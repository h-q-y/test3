import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment
import time
from Models.models import V2S, S2V, Att_update
from torch.optim import lr_scheduler
import random
import scipy.io as sio
import warnings
import logging
from Tools.DataLoader import Dataloader
from Tools.util_func import test_s2v, test_v2s, test_s2v_v2s
from timm.utils.log import setup_default_logging
warnings.filterwarnings('ignore')  
logging.getLogger().setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def BMVSc(a,b,device,n,m,opts):
    mask=list(range(n-m))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / (n-m) ## L2_Loss of seen classes
    A=a[n-m:]
    B=b[n-m:]
    DIS=torch.zeros((m,m))
    DIS=DIS.to(device)
    for A_id,x in enumerate(A):
        for B_id,y in enumerate(B):
            dis=((x-y)**2).sum()
            DIS[A_id,B_id]=dis
    matching_loss=0
    cost=DIS.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    for i,x in enumerate(row_ind):
        matching_loss+=DIS[row_ind[i],col_ind[i]]
    return L2_loss, matching_loss/m


if __name__=='__main__':


    parser=argparse.ArgumentParser(description="Training parameters setting")
    parser.add_argument('--dataset',type=str,default="AWA2", help='AWA2/SUN/CUB')
    parser.add_argument('--backbone',type=str,default="VIT", help='res101/VIT')

    parser.add_argument('--split_path',type=str,default="/home/shchen/projects/data/xlsa17/data/", help='the path of data split setting')
    parser.add_argument('--data_dir',type=str,default="/home/shchen/projects/zsl/code4cvpr2023/data/", help='the path of data')
    parser.add_argument('--VCCC_dir',type=str,default="/home/shchen/projects/zsl/code4cvpr2023/VC&CC/", help='the save path for cluster visual center ans real visual center')
    parser.add_argument('--output_dir',type=str,default="/home/shchen/projects/zsl/code4cvpr2023/results/")

    parser.add_argument('--finetune',default=False, help='choose use the visual features from finetune backbone or pretrained backbone')
    parser.add_argument('--method',type=str, default='BMVSc', help="We use BMVSc by default to determine the category corresponding to the cluster center")
    parser.add_argument('--GPU',type=str,default="0") 
    parser.add_argument('--lamda1',type=float,default=0.5, help="weight of BMVSc loss")
    parser.add_argument('--lamda2',type=float,default=1.0, help="weight of cross entropy loss for v2s model (use synthetic sample of s2v)")
    parser.add_argument('--lamda3',type=float,default=1.0, help="weight of cross entropy loss for v2s model (use real samples)")
    parser.add_argument('--alpha',type=float,default=0.9, help="control the predictive weights of the two networks(s2v ans v2s)")

    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--weight_decay',type=float,default=0.0005)
    parser.add_argument('--seed', default=1)

    parser.add_argument('--epochs',type=int,default=3000, help="training epochs")
    parser.add_argument('--eval_epochs',type=int,default=500, help="test the model every xx rounds")
    parser.add_argument('--is_balance',default=True, action='store_true', help="Control the dataloader to achieve category-balanced output of real samples, used for v2s")

    parser.add_argument('--update',default=False, action='store_true', help="use the att updata module or not(this mudule is useless now)")
    parser.add_argument('--update_begin',type=int,default=0)
    parser.add_argument('--update_ratio',type=float,default=0.01)
    args=parser.parse_args()

    #============Fix the random seed============#
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device=torch.device("cuda:"+args.GPU)
    args.device = device
    if args.dataset=="CUB":
        input_dim=312
        n=200
        m=50
    if args.dataset=="AWA2":
        input_dim=85
        n=50
        m=10
    if args.dataset=="SUN":
        input_dim=102
        n=717
        m=72

    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    if args.finetune == True:
        args.data_dir = args.data_dir+"ft_"+args.backbone+"_"+args.dataset+".mat"
        VC_dir = args.VCCC_dir+"ft_"+args.backbone+"_"+args.dataset+"_VC.mat"
        CC_dir = args.VCCC_dir+"ft_"+args.backbone+"_"+args.dataset+"_CC.mat"
        # set the save path of log file
        setup_default_logging(log_path=args.output_dir+args.dataset+"_"+time_str+'.log')
        logger.info("Training parameters %s", args)
    else:
        args.data_dir = args.data_dir+args.backbone+"_"+args.dataset+".mat"
        VC_dir = args.VCCC_dir+args.backbone+"_"+args.dataset+"_VC.mat"
        CC_dir = args.VCCC_dir+args.backbone+"_"+args.dataset+"_CC.mat"
        # set the save path of log file
        setup_default_logging(log_path=args.output_dir+args.dataset+"_"+time_str+'.log')
        logger.info("Training parameters %s", args)

    args.split_path = args.split_path+args.dataset+"/att_splits.mat"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    #============print the setting============#
    yy = vars(args)
    for k, v in sorted(yy.items()):
        print('%s: %s' % (str(k), str(v)))

    #============load the data============#
    matcontent = sio.loadmat(VC_dir)
    seen_class = matcontent["seen_class"][0].tolist()
    unseen_class = matcontent["unseen_class"][0].tolist()
    VC_seen = matcontent["seen_class_VC"]    #load the visual center of seen class
    VC_unseen = sio.loadmat(CC_dir)["unseen_class_CC"]   #load the cluster center of unseen class
    VC = np.concatenate([VC_seen, VC_unseen])
    VC = torch.tensor(VC).to(device)

    matcontent = sio.loadmat(args.split_path)
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    att = (matcontent['att'].T)[seen_class+unseen_class]
    att = torch.tensor(att).float().to(device)  #load the att

    matcontent = sio.loadmat(args.data_dir)
    features = matcontent['features'].T
    labels = matcontent['labels'].astype(int).squeeze() - 1
    dataloader = Dataloader(args)   # which is used to take real training data to train the V2S

    #============init the model============#
    if args.backbone == "VIT":
        output_dim=768
    else:
        output_dim=2048
    Net1 = S2V(input_dim, output_dim).to(device)
    Net2 = V2S(output_dim, input_dim).to(device)
    eyes = torch.eye(n).to(device)
    best_acc, epoch_1, best_H, epoch_2 = 0, 0, 0, 0

    #================training=====================#
    optimizer = torch.optim.Adam(list(Net1.parameters())+list(Net2.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    step_optim_scheduler = lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.1)

    for epoch in range(args.epochs + 1):
        Net1.train()
        Net2.train()
        step_optim_scheduler.step(epoch)

        ##=============================The original module, training the s2v=============================##
        syn_vc = Net1(att)
        loss1, loss2=BMVSc(syn_vc, VC, device,n,m,args)
        # Loss 1 is a regression loss on the seen class. By training this regression loss, 
        # the synthetic visual center of the seen class is closer to the real center of the seen class.
        # Loss 2 is corresponding to eq.6 (BMVSc) in the paper(Transductive Zero-Shot Learning with Visual Structure Constraint)
        
        ##===============================Training the v2s===============================##
        #----------Training using visual features synthesized by s2v----------#
        pred_att = Net2(syn_vc)
        S_pp = torch.einsum('ki,bi->bik',att, pred_att)
        S_pp = torch.sum(S_pp,axis=1) 
        loss3=nn.CrossEntropyLoss()(S_pp[:len(seen_class)], eyes[:len(seen_class)]) + (epoch/args.epochs)*nn.CrossEntropyLoss()(S_pp[len(seen_class):], eyes[len(seen_class):])

        #----------Training using real samples----------#
        # Here I set the number of real samples in each batch to be equal to the number of synthetic samples to train v2s, 
        # i.e. dataloader.batch_size=len(syn_vc)
        batch_label, batch_feature, batch_att = dataloader.next_batch()
        pred_att2 = Net2(batch_feature)
        S_pp2 = torch.einsum('ki,bi->bik',dataloader.att, pred_att2)
        S_pp2 = torch.sum(S_pp2,axis=1) 
        loss4=nn.CrossEntropyLoss()(S_pp2, batch_label)

        loss = loss1 + (epoch/args.epochs)*args.lamda1*loss2 + (epoch/args.epochs)*args.lamda2*loss3 + args.lamda3*loss4
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #============update Att============#
        ## this module is useless now
        if args.update != False and epoch>args.update_begin:
            ratio = (epoch/args.epochs)*args.update_ratio
            att = F.normalize((1-ratio)*att.detach() + ratio*(F.normalize(pred_att)).detach())

        #=====================testing=====================#
        if epoch%args.eval_epochs==0:
            logger.info("=======training epochs: %d=========="%(epoch))
            logger.info("Loss1 is %.5f | Loss2 is %.5f | Loss3 is %.5f | Loss4 is %.5f | Loss_all is %.5f | lr %.7f "%(loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss.item(),optimizer.param_groups[0]['lr']))

            ## (CZSL_acc, GZSL_S, GZSL_U, GZSL_H) is the result of the combine of s2v and v2s (args.alpha control the predictive weights of the two networks)
            ## (CZSL_acc_visual, GZSL_S_visual, GZSL_U_visual, GZSL_H_visual) is the result that only use s2v
            ## (CZSL_acc_sem, GZSL_S_sem, GZSL_U_sem, GZSL_H_sem) is the result that only use v2s
            CZSL_acc, GZSL_S, GZSL_U, GZSL_H, CZSL_acc_visual, GZSL_S_visual, GZSL_U_visual, GZSL_H_visual, CZSL_acc_sem, GZSL_S_sem, GZSL_U_sem, GZSL_H_sem = test_s2v_v2s(args, Net1, Net2, features, labels, test_seen_loc, test_unseen_loc, att, VC, seen_class, unseen_class, device)

            logger.info("CZSL result(syn): %.5f | CZSL result(visual): %.5f | CZSL result(sem): %.5f"%(CZSL_acc, CZSL_acc_visual, CZSL_acc_sem))
            logger.info("GZSL result(sys): S: %.5f | U: %.5f | H: %.5f"%(GZSL_S,GZSL_U,GZSL_H))
            logger.info("GZSL result(visual): S: %.5f | U: %.5f | H: %.5f"%(GZSL_S_visual,GZSL_U_visual,GZSL_H_visual))
            logger.info("GZSL result(sem): S: %.5f | U: %.5f | H: %.5f"%(GZSL_S_sem,GZSL_U_sem,GZSL_H_sem))
            if best_acc < max([CZSL_acc, CZSL_acc_sem, CZSL_acc_visual]):
                best_acc = max([CZSL_acc, CZSL_acc_sem, CZSL_acc_visual])
                epoch_1 = epoch
            if best_H < max([GZSL_H, GZSL_H_sem, GZSL_H_visual]):
                best_H = max([GZSL_H, GZSL_H_sem, GZSL_H_visual])
                epoch_2 = epoch

    logger.info("Best acc: %.5f, in epoch %d | Best H: %.5f, in epoch %d"%(best_acc,epoch_1,best_H,epoch_2))