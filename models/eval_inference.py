import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch

from dataloaders.sql_file import Sql_ops
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
import pickle
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE, SAVE_TOP, PROJECT_PATH
import dill as pkl
filepath = os.path.dirname(os.path.abspath(__file__))

def save_obj(save_dir, obj, name ):
    with open(save_dir+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def eval_rels(conf, run_mode=None,run_test=False,run_num =None, best_epoch=None):
    row_count = 0

    if SAVE_TOP: #save top 5 rels in db
        # write all rels to database
        db = Sql_ops()
        db.create_table()
        db.del_table('pred_rels_master')

    if run_mode is None:
        run_mode=conf.mode
        run_test = conf.test
        checkpt = conf.ckpt
    else:
        if run_num is None or best_epoch  is None:
            raise ValueError()
        checkpt = conf.save_dir+'/run'+str(run_num)+'/vgrel-'+str(best_epoch)+'.tar'

    from lib.context_model import TransformerModel

    #load correct dataset and loader
    if conf.dataset == 'vg':
        from dataloaders.visual_genome import VGDataLoader, VG
    elif conf.dataset == 'gqa':
        from dataloaders.gqa import VGDataLoader, VG
    else:
        raise ValueError("Please mention a dataset")

    train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                                 use_proposals=conf.use_proposals, o_valid=conf.o_valid,
                                 filter_non_overlap=conf.mode == 'sgdet', require_overlap=conf.require_overlap_det)
    if run_test:
        val = test

    train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size, num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus, vg_mini=conf.vg_mini, o_val=conf.o_valid, shuffle=False)

    detector = TransformerModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates, config=conf, filenames= val.filenames ,inference=True)

    detector.cuda()
    ckpt = torch.load(checkpt, map_location='cpu')
    special_case =False #change the weight as per relation predition module
    if special_case:
        for n in ckpt['state_dict'].copy().keys():
            if n.startswith(('sub_obj_emb', 'final_rel', 'e_pos_emb')):
                ckpt['state_dict']['rel_prediction.' + n] = ckpt['state_dict'][n]

    optimistic_restore(detector, ckpt['state_dict'])
    del ckpt  #free up the memory of cuda

    def inference_batch(batch_num, b, infer_dict, row_count, dataset, eval_train=False):
        det_res = detector[b,eval_train]
        if conf.num_gpus == 1:
            det_res = [det_res]
        for i, (boxes_i, objs_i, obj_scores_i, attr_dist, rels_i, gt_rels_ind, pred_scores_i, obj_preds, p_o_r_m, g_o_r_m, pred_edge, true_edge) in enumerate(det_res[0]):
            file_name = dataset.filenames[batch_num*conf.batch_size + i]
            gt_classes = dataset.gt_classes[batch_num*conf.batch_size + i].copy()
            gt_relations = dataset.relationships[batch_num*conf.batch_size + i].copy()
            gt_boxes = dataset.gt_boxes[batch_num*conf.batch_size + i].copy()
            gt_attr = dataset.gt_attr[batch_num*conf.batch_size + i].copy()

            no_of_gt_class = len(gt_classes)   #to get rid of padding
            assert np.max(rels_i) <= no_of_gt_class - 1

            pred_boxes = (boxes_i * BOX_SCALE / IM_SCALE)[:no_of_gt_class,:]
            pred_classes = objs_i[:no_of_gt_class]
            pred_rel_inds = rels_i
            gt_rel_inds = gt_rels_ind
            obj_scores = obj_scores_i[:no_of_gt_class]
            rel_scores = pred_scores_i  # hack for now.
            obj_preds = obj_preds[:no_of_gt_class]
            pred_attr = attr_dist

            assert len(pred_classes)== len(gt_classes)

            #save network output
            infer_dict[file_name] = {'gt_cls': gt_classes, 'gt_rels':gt_relations, 'gt_boxes':gt_boxes,
                                     'obj_score':obj_scores, 'rel_scores': rel_scores, 'pred_rels':pred_rel_inds, 'gt_attr': gt_attr, 'pred_attr':pred_attr}

            row_count+=1
        return row_count
    # todo replace wd proper param
    detector.eval()
    #run inference for training
    save_dir  = os.path.join(PROJECT_PATH,'data', conf.dataset, 'infer_out')
    infer = {}
    print('Starting to save network inference in : ',save_dir)
    # for train_b, batch in enumerate(tqdm(train_loader)):
    #     row_count = inference_batch(conf.num_gpus * train_b, batch, infer, row_count, train, eval_train=True)
    # print("Number of training instances saved : ",row_count)
    # save_obj(save_dir, infer, 'infer_train'+conf.run_desc, )
    #run inference for validation
    infer ={}
    row_count = 0 #reset row count
    for val_b, batch in enumerate(tqdm(val_loader)):
            row_count = inference_batch(conf.num_gpus*val_b, batch, infer, row_count, val)
    print("Number of validation instances saved : ", row_count)
    save_obj(save_dir, infer, 'infer_val'+conf.run_desc,)


if __name__ == "__main__":
    conf = ModelConfig(file=os.path.join(filepath, 'param.txt'))  # write all param to file
    conf.mode='predcls'
    eval_rels(conf)
