"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
#set seed fro testing
from lib.helper import get_params,seed_torch
#seed_torch(42)

from torch import optim
import pandas as pd
import time
import sys
#import adabound
from lib.AdamW import AdamW,RAdam
import multiprocessing
filepath = os.path.dirname(os.path.abspath(__file__))
from config import ModelConfig, BOX_SCALE, IM_SCALE,DATA_PATH,VIDEO_DICT_PATH

from torch.nn import functional as F
import torch.nn as nn
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tensorboardX import SummaryWriter
from lib.pytorch_misc import print_para
from models.eval_rels import eval_rels
from lib.FocalLoss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
#from warmup_scheduler import GradualWarmupScheduler

conf = ModelConfig(file = os.path.join(filepath,'param.txt')) #write all param to file

if conf.dataset == 'vg':
    from dataloaders.visual_genome import VGDataLoader, VG
elif conf.dataset == 'gqa':
    from dataloaders.gqa import VGDataLoader, VG
else:
    raise ValueError ("Please mention a dataset")

if conf.model == 'motifnet':
    from lib.context_model import TransformerModel

#filename_to_save = 'lr_'+str(conf.lr)+'_nl_obj'+str(conf.nl_obj)+'_nl_edge'+str(conf.nl_edge)+'_fwd_dim'+str(conf.fwd_dim)+'_attn_dim'+str(conf.attn_dim)
if conf.spatial_box:
    assert  not conf.normalized_roi, 'Spatial Box and Normalized Box cant be True same time'
writer = SummaryWriter(comment='_run#'+str(conf.run_desc))


train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                              use_proposals=conf.use_proposals,o_valid = conf.o_valid,
                              filter_non_overlap=conf.mode == 'sgdet', require_overlap = conf.require_overlap_det)
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size, num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus, vg_mini =conf.vg_mini, o_val =conf.o_valid )

if conf.count_e_dist:
    pos_edge = []
    neg_edge = []
if conf.use_FL:
    FL = FocalLoss(num_class=len(train.ind_to_predicates), alpha=0.1, gamma=2.0, balance_index=2)

detector = TransformerModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates, config=conf)

assert not (conf.freeze_obj_enc == True and conf.reduce_lr_obj_enc == True), 'Both freeze and reduce lr cant be set same time for obj_enc'
assert not (conf.train_obj_roi==True and conf.freeze_obj_enc==True), 'train obj roi cant be true if obj enc freezed'
# # Freeze the detector
for n, param in detector.named_parameters():
    if n.startswith(('detector','roi_extractor')) and  not conf.train_detector:
        param.requires_grad = False
    if n.startswith(('roi_fmap', 'context.decoder_lin')) and  not conf.train_obj_roi:  #freeze the object classifiar:
        param.requires_grad = False
    if n.startswith(('context.obj_ctx_enc', 'context.compress_node', 'context.pos_embed.0', 'context.pos_embed.1'))  and conf.freeze_obj_enc:  # freeze the object classifiar
        param.requires_grad = False
#print(print_para(detector), flush=True)


def get_optim(lr):

    params = get_params(conf, detector, lr)

    if conf.optimizer =='adam':
        optimizer = optim.Adam(params, lr=conf.lr,betas=(0.9, 0.98)) #weight_decay=conf.l2,
    # elif conf.optimizer == 'adabound':
    #     optimizer = adabound.AdaBound(params, lr=conf.lr, final_lr=0.1)
    elif conf.optimizer == 'radam':
        optimizer = RAdam(params, lr=conf.lr, weight_decay=conf.l2)
    elif conf.optimizer == 'adamw':
        optimizer = AdamW(params, lr=conf.lr, weight_decay=conf.l2)
    elif conf.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=conf.lr, weight_decay=conf.l2, momentum=0.9)
    else:
        print('Using SGD')
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.8,
                                   verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)
    #scheduler = StepLR(optimizer,11,0.1)
    #scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=2, total_epoch=15, after_scheduler=scheduler)

    return optimizer, scheduler, None #scheduler_warmup

if conf.use_mmdet:
    if conf.ckpt:
        # ckpt = torch.load(conf.ckpt)
        # optimistic_restore(detector, ckpt['state_dict'])
        start_epoch = -1
    else:
        start_epoch = -1
elif len(os.path.basename(conf.ckpt).split('-')) == 2 and not conf.dataset=='gqa':  #todo set start epoch
    ckpt = torch.load(conf.ckpt)
    if detector.training:
        print("Loading Detector, ROI and Object Classifier")
        for n in ckpt['state_dict'].copy().keys():
            if n.startswith(('post_lstm','rel_compress','union_boxes')): # or n.startswith('freq_bias')
                del (ckpt['state_dict'][n])
            if n.startswith('context') and not n.startswith('context.pos_embed'):
                del (ckpt['state_dict'][n])

        if not optimistic_restore(detector, ckpt['state_dict']):
            start_epoch = -1
        print('Loaded Detector, Fmap and Freq baseline')
    else:
        start_epoch = -1

    print('Loaded Detector, Fmap and Freq baseline')
else:
    ckpt = torch.load(conf.ckpt)
    start_epoch = -1
    #if conf.dataset =='gqa':
        # for n in ckpt['state_dict'].copy().keys():
        #         # if n.startswith(('backbone')):  # or n.startswith('freq_bias')
        #         #     layer_name = n.split('.',1)
        #         layer_name='detector.'+n #layer_name[1]
        #         ckpt['state_dict'][layer_name] = ckpt['state_dict'].pop(n)

    optimistic_restore(detector.detector, ckpt['state_dict'])
    if conf.union_boxes:
        detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])   # for viusal rep union box related
        detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])    # to get the object feature map from vgg
    detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    print('Loaded Detector and Fmap')

detector.cuda()


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch , epoch_num*len(train_loader)+b, verbose=b % (conf.print_interval*10) == 0)) #b == 0))
        if b % conf.print_interval == 0 :
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('Learning rate :',[p_lr['lr'] for p_lr in optimizer.param_groups])
            #tensorboard output
            writer.add_scalar('data/class_loss', mn.class_loss, (epoch_num*len(train_loader)+b))
            writer.add_scalar('data/rel_loss', mn.rel_loss, (epoch_num*len(train_loader)+b))
            if conf.use_attr: #for attribute prediction
                writer.add_scalar('data/attr_loss', mn.attr_loss, (epoch_num * len(train_loader) + b))
            if conf.use_obj_rel_map:
                writer.add_scalar('data/obj_rel_loss', mn.obj_rel_loss, (epoch_num * len(train_loader) + b))
            if conf.use_valid_edges:
                writer.add_scalar('data/edge_loss', mn.edge_loss, (epoch_num * len(train_loader) + b))
            writer.add_scalar('data/total_loss', mn.total, (epoch_num * len(train_loader) + b))

            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b, itr, verbose=False,):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}
    losses['class_loss'] = conf.g1*F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    if conf.use_attr:
        bce = nn.BCELoss()  #init binary cross entropy loss
        losses['attr_loss'] = bce(result.attr_dist, result.od_attrs[:,2:].float())
    if conf.use_FL:
        losses['rel_loss'] = conf.g2*FL(result.rel_dists, result.all_rels[:, -1])
    elif conf.reduce_bg_loss:
        idx_bg = torch.squeeze(torch.nonzero(result.all_rels[:, -1] == 0))
        idx_fg = torch.squeeze(torch.nonzero(result.all_rels[:, -1]))
        M_FG = len(idx_fg)
        M_BG = len(idx_bg)
        M = len(result.rel_dists)
        edge_weights = torch.ones(M, device=result.all_rels.get_device())
        if M_FG > 0:
            edge_weights[idx_fg] = float(conf.g2) / M_FG
        if M_BG > 0:
            edge_weights[idx_bg] = float(conf.g3) / M_BG  # weight for BG edges (beta/M_BG instead of 1/M as in the baseline)

        bg_fg_loss = F.cross_entropy(result.rel_dists, result.all_rels[:, -1], reduction='none')
        loss = conf.g4 * bg_fg_loss * torch.autograd.Variable(edge_weights)
        #losses['rel_loss'] = conf.g2 * torch.sum(bg_fg_loss[bg] / len(bg)) + conf.g3 * torch.sum(bg_fg_loss[fg] / len(fg))
        losses['rel_loss'] = loss.sum()
    else:
        losses['rel_loss'] = conf.g2 * F.cross_entropy(result.rel_dists, result.all_rels[:, -1])
    if conf.use_obj_rel_map:
        losses['obj_rel_loss'] = conf.g3*F.binary_cross_entropy(result.pred_obj_rel_mat.view(-1), result.gt_obj_rel_mat.view(-1))
    if conf.count_e_dist:
        pos_edge.append(torch.nonzero(result.all_rels[:, -1]).shape[0])  # result.edge_gt_lbl.sum().data.cpu().numpy()
        neg_edge.append(len(result.all_rels[:, -1]) - torch.nonzero(result.all_rels[:, -1]).shape[
            0])  # len(result.edge_gt_lbl)-result.edge_gt_lbl.sum().data.cpu().numpy()
    if conf.use_valid_edges:

        losses['edge_loss'] = conf.g4*F.binary_cross_entropy(result.edge_dist.view(-1), result.edge_gt_lbl.float())


    loss = sum(losses.values())


    optimizer.zero_grad()
    loss.backward()
    #ploat gradients apart from transformer
    if verbose:
        for n, p in detector.named_parameters():
            if not n.startswith(('context.obj_ctx','context.edge_ctx',)) and p.requires_grad and p.grad !=None:
                writer.add_histogram(n, p.grad, itr)
    if conf.optimizer=='sgd':
        clip_grad_norm(
            [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
            max_norm=conf.clip, verbose=verbose, clip=True)    #todo experiemnt without clip
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data.cpu() for x, y in losses.items()})
    return res


def val_epoch(epoch):
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats( epoch, writer)
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100]), np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][20])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]
    for i, (boxes_i, objs_i, obj_scores_i, attr_dist, rels_i, gt_rels_ind, pred_scores_i, obj_preds, p_o_r_m, g_o_r_m, pred_edge, true_edge) in enumerate(det_res[0]):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num*conf.batch_size + i].copy(),
            'gt_relations': val.relationships[batch_num*conf.batch_size + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num*conf.batch_size + i].copy(),
            'gt_attrs': val.gt_attr[batch_num*conf.batch_size + i][:,1:].copy() if conf.dataset == 'gqa' else None,
        }
        #assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
        no_of_gt_class = len(gt_entry['gt_classes'])   #to get rid of padding
        assert np.max(rels_i) <= no_of_gt_class - 1

        pred_entry = {
            'pred_boxes': (boxes_i * BOX_SCALE / IM_SCALE)[:no_of_gt_class,:],
            'pred_classes': objs_i[:no_of_gt_class],
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i[:no_of_gt_class],
            'pred_attrs': attr_dist[:,1:] if conf.use_attr else None,
            'rel_scores': pred_scores_i,  # hack for now.
            'obj_preds': obj_preds[:no_of_gt_class],
            'p_o_r_m': p_o_r_m,
            'g_o_r_m' : g_o_r_m,
            'pred_edge':pred_edge,
            'true_edge':true_edge,

        }
        assert len(pred_entry['pred_classes'])== len(gt_entry['gt_classes'])

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )

print("Training starts now!")
optimizer, scheduler, scheduler_warmup = get_optim(conf.lr ) #* conf.num_gpus * conf.batch_size
#save best two
best_epochs = 0 #{}
best_epoch_no = 0
#mAp = 0
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):

    # mAp, mAp20 = val_epoch(epoch) #only for testing validation
    # sys.exit()
    #scheduler_warmup.step(epoch, mAp)
    rez = train_epoch(epoch)
    if conf.count_e_dist:
        print('Total pos edge : ',sum(pos_edge), ' and neg edge : ', sum(neg_edge))
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    mAp, mAp20 = val_epoch(epoch)

    # save only two best epoch or best epoch
    #if len(best_epochs)==0 or mAp20 > min(best_epochs.values()):
    if best_epochs == 0 or best_epochs <= mAp20:
        if conf.save_dir is not None:
            save_dir = conf.save_dir
        if conf.run_desc !=0:
            save_dir = os.path.join(save_dir, 'run' + str(conf.run_desc))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        #if len(best_epochs)==2:
            #min_key = min(best_epochs, key=best_epochs.get)
        if best_epochs >0:
            try:
                os.remove(os.path.join(save_dir, '{}-{}.tar'.format('vgrel', best_epoch_no))) #min_key
            except Exception:
                print('Not able to find checkpoint of ',best_epoch_no)#min_key
            #del best_epochs[min_key]
        #best_epochs[epoch]=mAp20
        best_epochs = mAp20
        best_epoch_no = epoch


        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(save_dir, '{}-{}.tar'.format('vgrel', epoch)))
    scheduler.step(mAp)


writer.close()
#now run evaluation on test
for eval_mode in('predcls','sgcls'):
    eval_rels(conf, run_mode=eval_mode, run_test=True, run_num=conf.run_desc, best_epoch=best_epoch_no)


