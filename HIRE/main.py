import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import time
import warnings
import argparse
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import numpy as np
import utils
import shutil
from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature_ours, tsne, a_distance
from scdataset import Prepare_scDataloader
from scmodel import sc_net
from cluster import ClusterLoss, BCE, PairEnum
from utils import get_ulb_sim_matrix, curriculum_scheduler, CSOT_PL,mask_generate,SupConLoss, Difficulty,lws_loss,confidence_update_lw
from entropy import TsallisEntropy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")



def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        #torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    labeled_train_source_loader,unlabeled_train_source_loader, test_source_loader, train_target_loader, test_target_loader, gene_size, type_num = Prepare_scDataloader(args).getloader()

    
    labeled_train_source_iter = ForeverDataIterator(labeled_train_source_loader)
    unlabeled_train_source_iter = ForeverDataIterator(unlabeled_train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    diff = Difficulty(type_num, accumulate_iters=50)
    # encoder = sc_encoder(gene_size).cuda()
    # classifier = sc_classifier(type_num).cuda()
    classifier = sc_net(gene_size,type_num).cuda()
    
    # define optimizer and lr scheduler
    optimizer = SGD(classifier.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier).to(device)
        args.label_ratio = 1
        labeled_train_source_loader,_, _, _, _, _, _ = Prepare_scDataloader(args).getloader()
        source_feature = collect_feature_ours(labeled_train_source_loader, feature_extractor, device)
        target_feature = collect_feature_ours(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1, novel_samples = utils.validate(test_target_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    # x_train = labeled_train_source_loader.dataset.data_reader
    # labels = labeled_train_source_loader.dataset.labels
    # labeld_length = len(labeled_train_source_loader.dataset)
    # unlabeled_length = len(unlabeled_train_source_loader.dataset)
    # labels[labeld_length:] = -1
    # lp_model = LabelPropagation(gamma=20,max_iter=1000)
    # lp_model.fit(x_train, labels)
    # y_pred = lp_model.predict(x_train)
    # labeled_train_source_loader.dataset.labels = pred
    
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())
        # train for one epoch
        if epoch < args.warm_up:
            print(f"warm up epoch: {epoch}/{args.warm_up}")
            warmup(labeled_train_source_iter, classifier, optimizer, lr_scheduler, epoch,type_num, args)
            
        #elif epoch == args.warm_up:
        else:
            
            
            
            #lp_model = LabelPropagation(gamma=20,max_iter=1000)
            # lp_model.fit(x_train, y_mixed)
            # y_pred = lp_model.predict(x_train)
            
            budget, pho = curriculum_scheduler(epoch-args.warm_up, args.epochs-args.warm_up, 
                                    begin=args.begin_rate, end=1, mode='linear')
            print(f"current budget = {budget} ({pho*100}%)")
            
            with torch.no_grad():
                all_pseudo_labels, all_argmax_plabels,  all_conf,all_gt_labels = CSOT_PL(classifier, unlabeled_train_source_loader, num_class=type_num, batch_size=args.batch_size, 
                                                                                            budget=budget)
                all_pseudo_labels, all_argmax_plabels, all_conf,all_gt_labels = all_pseudo_labels.cpu(),all_argmax_plabels.cpu(),all_conf.cpu(),all_gt_labels.cpu()
                # print('all_pseudo_labels:',all_pseudo_labels[:10])
                # print('all_argmax_plabels:',all_argmax_plabels[:10])
                # print('all_gt_labels:',all_gt_labels[:10])
                # print('all_conf:',all_conf[:10])
                
                # sys.exit()
                
                labeld_length = len(labeled_train_source_loader.dataset)
                assert labeld_length + len(all_argmax_plabels) == len(unlabeled_train_source_loader.dataset.labels)
                unlabeled_train_source_loader.dataset.labels[labeld_length:] = np.array(all_argmax_plabels)
                unlabeled_train_source_iter = ForeverDataIterator(unlabeled_train_source_loader)
                args.label_ratio = 1
                labeled_source_loader,_, _, _, _, _, _ = Prepare_scDataloader(args).getloader()
                labeled_source_iter = ForeverDataIterator(labeled_source_loader)
        #else:   

            train(labeled_source_iter,unlabeled_train_source_iter, train_target_iter, classifier, optimizer, lr_scheduler, epoch, type_num,diff,train_target_loader, args)

        # evaluate on validation set
        #acc1 = utils.validate(test_source_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        #torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        #if acc1 > best_acc1:
        #    shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        #best_acc1 = max(acc1, best_acc1)

    #print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    #classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
            acc1  = utils.validate(test_target_loader, classifier, args, device)
            torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
            print("test_acc1 = {:4.2f}".format(acc1))
            if best_acc1 < acc1:
                best_acc1 = acc1
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            print("best_acc1 = {:4.2f}".format(best_acc1))
        
    print("best_acc: {:4.2f}".format(best_acc1))
    

    
    logger.close()


    
def train(labeled_train_source_iter: ForeverDataIterator, unlabeled_train_source_iter: ForeverDataIterator,train_target_iter: ForeverDataIterator,
          model: sc_net, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, type_num: int, diff,train_target_loader, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Total Loss', ':6.2f')
    un_losses = AverageMeter('Un Loss', ':6.2f')
    sf_losses = AverageMeter('Sf Loss', ':6.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [ losses,un_losses,sf_losses],
        prefix="Epoch: [{}]".format(epoch))

    # atac_dataset = train_target_loader.dataset.data_reader
    # scaler = StandardScaler()
    # atac_data_scaled = scaler.fit_transform(atac_dataset)
    # n_components = 227  
    # pca = PCA(n_components=n_components)
    # atac_data_reduced = pca.fit_transform(atac_data_scaled)   
    # knn_ind = NN(atac_data_reduced, query=atac_data_reduced, k=11, metric='manhattan', n_trees=10)[:, 1:]
    # knn_ind = knn_ind.astype('int64')
    
    n, c = len(train_target_loader.dataset), type_num
    confidence = torch.ones(n, c) / c
    confidence = confidence.to(device)
    
    
    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold).to(device)
    cluster_loss = ClusterLoss(device, type_num, "cos", args.cos_distance_threshold, args.topk)
    ts = TsallisEntropy(temperature=args.temperature, alpha=args.alpha)
    bce = BCE()
    criterion_simclr = SupConLoss()
    back_cluster = True if epoch >= args.back_cluster_start_epoch else False
    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(labeled_train_source_iter)[:2]
        x_t, labels_t,index = next(train_target_iter)[:3]
        x_s_strong = x_s
        x_t_strong = x_t
        x_s = add_noise(x_s).to(device)
        x_t = add_noise(x_t).to(device)
        x_s_strong = add_noise(x_s_strong).to(device)
        x_t_strong = add_noise(x_t_strong).to(device)
        # x_s = x_s.to(device)
        # x_t = x_t.to(device)
        # x_s_strong = x_s_strong.to(device)
        # x_t_strong = x_t_strong.to(device)
        
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        index = index.to(device)
        
        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()

        # compute output
        #with torch.no_grad():
        f_t,y_t,cluster_t,eqinv_t = model(x_t)

        # cross entropy loss
        f_s,y_s,cluster_s,eqinv_s = model(x_s)
        weight_diff = diff.cal_weights(y_s.detach(), labels_s)
        
        cls_loss = F.cross_entropy(y_s, labels_s,weight=weight_diff)

        soft_pseudo_labels = F.softmax(y_t.detach(), dim=1)
        confidence1, _ = soft_pseudo_labels.max(dim=1)
        soft_pseudo_labels = soft_pseudo_labels[confidence1 < args.threshold]
        Y = torch.zeros_like(soft_pseudo_labels )
        values,indices = torch.topk(soft_pseudo_labels, k = args.pll_topk,dim=1)
        Y.scatter_(1,indices,1)
 
        if len(soft_pseudo_labels)>0:
            #print('outputs:',y_t[confidence1 < args.threshold].shape,'Y:',Y.shape)
            #print(Y[0])
            un_loss, _, _ = lws_loss(y_t[confidence1 < args.threshold], Y.float(), confidence, index[confidence1 < args.threshold], 1, 1, None)
        else:
            un_loss = torch.from_numpy(np.array(0)).to(device)
        # self-training loss
        f_s_strong,y_s_strong,cluster_s_strong,eqinv_s_strong = model(x_s_strong)
        f_t_strong,y_t_strong,cluster_t_strong,eqinv_t_strong = model(x_t_strong)
        self_training_loss, mask, pseudo_labels = self_training_criterion(y_t_strong, y_t)
        self_training_loss = args.trade_off * self_training_loss
        
        
        
        labels_t_scrambled = torch.ones_like(labels_t).to(device) + cluster_loss.num_classes
        ######### EqInv loss
        preds1_u = torch.cat((eqinv_s.detach(), eqinv_t.detach()), dim=0)
        preds2_u = torch.cat((eqinv_s.detach(), eqinv_t.detach()), dim=0)
        inputs = {
            "x1": torch.cat((f_s, f_t), dim=0),
            "preds1_u": preds1_u,
            "preds2_u": preds2_u,
            "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
        }
        bce_loss, _ = cluster_loss.compute_losses(inputs)   # Cluster loss (for EqInv cluster head if applicable)
        
        p_t_nograd = F.softmax(y_t.detach(), dim=1)
        p_t_u_nograd = F.softmax(y_t_strong.detach(), dim=1)
        p_s_alt_nograd = F.softmax(cluster_s.detach(), dim=1)
        p_s_u_alt_nograd = F.softmax(cluster_s_strong.detach(), dim=1)
        y_alt2 = torch.cat((cluster_s,cluster_t),dim=0)
        y_alt2_nograd = torch.cat((cluster_s.detach(),cluster_t.detach()),dim=0)
        y_u_alt2 = torch.cat((cluster_s_strong,cluster_t_strong),dim=0)
        y_u_alt2_nograd = torch.cat((cluster_s_strong.detach(),cluster_t_strong.detach()),dim=0)
        inputs = {
            "x1": torch.cat((f_s, f_t), dim=0),
            "preds1_u": y_alt2 if back_cluster else y_alt2_nograd,
            "preds2_u": y_u_alt2 if back_cluster else y_u_alt2_nograd,
            "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
        }
        ######## Cluster loss (for target domain clustering)
        bce_loss_u, sim_matrix_ulb = cluster_loss.compute_losses(inputs)
        bce_loss += bce_loss_u
        y_t_alt2 = cluster_t
        #y_t_u_alt2 = cluster_t_strong
        #max_prob_alt2, pseudo_labels_alt2 = torch.max(F.softmax(y_t_alt2, dim=1), dim=-1)
        # st_loss_cluster = (F.cross_entropy(y_t_u_alt2, pseudo_labels_alt2,
        #                     reduction='none') * max_prob_alt2.ge(args.threshold).float().detach()).mean()
        # Refine unlabel similarity matrix (filter out uncertain pairs)
        cluster_logits = y_t_alt2
        sim_matrix_ulb_refined, low_t, high_t = get_ulb_sim_matrix(
            args.con_mode, sim_matrix_ulb, cluster_logits,
        )
        # classification head consistent with u clusters
        ## pt_nograd ## B x C
        weight = model.classifier[1].weight.detach()   ## C x D
        f_target = torch.matmul(p_t_nograd, weight)
        feat_row, feat_col = PairEnum(F.normalize(f_target, dim=1))
        tmp_distance_ori = torch.bmm(
            feat_row.view(feat_row.size(0), 1, -1),
            feat_col.view(feat_row.size(0), -1, 1)
        )
        tmp_distance_ori = tmp_distance_ori.squeeze()
        u_sim_matrix = torch.zeros_like(tmp_distance_ori).float() - 1
        u_sim_matrix[tmp_distance_ori > args.cos_distance_threshold] = 1
        
        pairs1, _ = PairEnum(p_t_nograd)
        _, pairs2 = PairEnum(p_t_u_nograd)
        con_loss_u = bce(pairs1, pairs2, u_sim_matrix)
        #con_loss_u = bce(pairs1, pairs2, sim_matrix_ulb_refined)
        # cluster head consistent with s labels (to improve clustering)
        labels_s_view = labels_s.contiguous().view(-1, 1)
        sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(device)
        sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
        pairs1, _ = PairEnum(p_s_alt_nograd)
        _, pairs2 = PairEnum(p_s_u_alt_nograd)
        con_loss_s = bce(pairs1, pairs2, sim_matrix_lb.flatten())

        ########### Consistency loss
        con_loss = con_loss_u + con_loss_s

        ########### Invariant loss
        sim_matrix_ulb_full, _, _ = get_ulb_sim_matrix(
            'stats', sim_matrix_ulb, cluster_logits, update_list=(args.con_mode=='stats')
        )   # get full ulb pairwise labels for invariant loss
        p_t, p_t_u = F.softmax(y_t, dim=1), F.softmax(y_t_strong, dim=1)
        pairs1, _ = PairEnum(p_t)
        _, pairs2 = PairEnum(p_t_u)
        #irm_con_t = bce(pairs1, pairs2, sim_matrix_ulb_full)
        irm_con_t = bce(pairs1, pairs2, u_sim_matrix)
        p_s, p_s_u = F.softmax(y_s, dim=1), F.softmax(y_s_strong, dim=1)
        pairs1, _ = PairEnum(p_s)
        _, pairs2 = PairEnum(p_s_u)
        irm_con_s = bce(pairs1, pairs2, sim_matrix_lb.flatten())
        inv_loss = torch.var(torch.stack([irm_con_t, irm_con_s]))
        
        
        transfer_loss = ts(y_t)
        
        
        # loss_ulb_0, ft_rank = ulb_rank(f_t_strong, 2)
            
        # loss_ulb_1 = ulb_rank_prdlb(y_t_strong, 2, pred_inp=ft_rank)
        
        # measure accuracy and record loss
        loss = transfer_loss+ cls_loss + self_training_loss + bce_loss+ con_loss + inv_loss + 0.1*un_loss
        loss.backward()
        losses.update(loss.item(), x_s.size(0))
        un_losses.update(un_loss.item(), x_s.size(0))
        sf_losses.update(self_training_loss.item(), x_s.size(0))
  
 
        # compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()
        
        if len(soft_pseudo_labels)>1:
            confidence = confidence_update_lw(model, confidence, x_t[confidence1 < args.threshold], Y, index[confidence1 < args.threshold])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def warmup(labeled_train_source_iter: ForeverDataIterator, model: sc_net, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, type_num: int, args: argparse.Namespace):

    model.train()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(labeled_train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # clear grad
        optimizer.zero_grad()

        # cross entropy loss
        f_s,y_s,cluster_s,eqinv_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)

        loss =  cls_loss
        loss.backward()
        # compute gradient and do SGD step
        optimizer.step()
        #lr_scheduler.step()

            
def add_noise(data, noise_level=0.001):
    noise = noise_level * torch.randn_like(data)
    return data + noise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FixMatch for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--rna_data', default= ['/home/ai4science/scJoint/data/citeseq_control_rna.npz'])
    parser.add_argument('--rna_label', default= ['/home/ai4science/scJoint/data/citeseq_control_cellTypes.txt'])
    parser.add_argument('--atac_data', default= ['/home/ai4science/scJoint/data/asapseq_control_atac.npz'])
    parser.add_argument('--atac_label', default= ['/home/ai4science/scJoint/data/asapseq_control_cellTypes.txt'])
    parser.add_argument('--dataset', default= 'scRNA_10X_v2-ATAC')
    parser.add_argument('--label_ratio', type=float, default= 0.1 )
    
    # model parameters
    parser.add_argument('--trade-off', default=1., type=float,help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=32, type=int, help='mini-batch size of unlabeled data (target domain) (default: 32)')
    parser.add_argument('--threshold', default=0.9, type=float, help='confidence threshold')
    parser.add_argument('--cos_distance_threshold', default=0.8, type=float, help='acos_distance_threshold')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0004, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--epochs', default=60, type=int, metavar='N')
    parser.add_argument('--warm_up', default=30, type=int, metavar='N')
    parser.add_argument('--begin_rate', default=0.3, type=float) 
    parser.add_argument('--back_cluster_start_epoch', default=100, type=int, help='starting epoch to back cluster loss')
    parser.add_argument('--pll_topk', default=4, type=int, help='candidate partial labels for uncertain samples')
    parser.add_argument('--topk', default=5, type=int, help='rank statistics threshold for clustering')
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling for entropy')
    parser.add_argument('--alpha', default= 1.9, type=float, help='the entropic index of Tsallis loss')
    parser.add_argument('--con_mode', type=str, default='stats', help='gt | stats | sim')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,  metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',  help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='ours_atac', help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    
    if args.dataset =='cite-asap':
        args.rna_data = ['/home/ai4science/scJoint/data/citeseq_control_rna.npz'] 
        args.rna_label = ['/home/ai4science/scJoint/data/citeseq_control_cellTypes.txt'] 
        args.atac_data = ['/home/ai4science/scJoint/data/asapseq_control_atac.npz'] 
        args.atac_label = ['/home/ai4science/scJoint/data/asapseq_control_cellTypes.txt']  
    elif args.dataset == "scRNA_SMARTer-ATAC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "scRNA_SMARTer-snmC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_SMARTer_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "snRNA_10X_v3_A-ATAC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "snRNA_10X_v3_A-snmC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v3_A_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "snRNA_10X_v2-ATAC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_cellTypes.txt']           
    elif args.dataset == "snRNA_10X_v2-snmC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v2_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_cellTypes.txt']            
    elif args.dataset == "snRNA_SMARTer-ATAC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_cellTypes.txt'] 
    elif args.dataset == "snRNA_SMARTer-snmC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_SMARTer_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_snRNA_SMARTer_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v3-ATAC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v3-snmC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v3_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v3_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v2-ATAC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_ATAC_cellTypes.txt']
    elif args.dataset == "scRNA_10X_v2-snmC":
        args.rna_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v2_exprs.npz']
        args.rna_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_RNA_scRNA_10X_v2_cellTypes.txt']
        args.atac_data = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_exprs.npz']
        args.atac_label = ['/home/ai4science/scJoint/data_MOp/YaoEtAl_snmC_cellTypes.txt']           
    main(args)
