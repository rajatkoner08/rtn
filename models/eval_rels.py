import sys
import os
sys.path.append(os.getcwd())


from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch

from dataloaders.sql_file import Sql_ops
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator,calculate_mR_from_evaluator_list, eval_entry
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE, SAVE_TOP
import dill as pkl
filepath = os.path.dirname(os.path.abspath(__file__))

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

    train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                              use_proposals=conf.use_proposals,o_valid = conf.o_valid,
                              filter_non_overlap=run_mode == 'sgdet', require_overlap = conf.require_overlap_det)
    if run_test:
        val = test
    train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size, num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus, vg_mini =conf.vg_mini, o_val =conf.o_valid )
    #todo remove this constraint if more GPU are there
    #torch.cuda.set_device(1)
    detector = TransformerModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates, config=conf, filenames= val.filenames )

    detector.cuda()
    ckpt = torch.load(checkpt,map_location='cpu')

    special_case =True #change the weight as per relation predition module
    if special_case:
        for n in ckpt['state_dict'].copy().keys():
            if n.startswith(('sub_obj_emb', 'final_rel', 'e_pos_emb')):
                ckpt['state_dict']['rel_prediction.' + n] = ckpt['state_dict'][n]

    optimistic_restore(detector, ckpt['state_dict'])
    # if run_mode == 'sgdet':
    #     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
    #     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
    #     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
    #     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
    #     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

    all_pred_entries = []

    def val_batch(batch_num, b, evaluator, row_count, thrs=(20, 50, 100)):
        det_res = detector[b]
        if conf.num_gpus == 1:
            det_res = [det_res]
        for i, (boxes_i, objs_i, obj_scores_i, rels_i, gt_rels_ind, pred_scores_i, obj_preds, p_o_r_m, g_o_r_m, pred_edge, true_edge) in enumerate(det_res[0]):
            gt_entry = {
                'gt_classes': val.gt_classes[batch_num*conf.batch_size + i].copy(),
                'gt_relations': val.relationships[batch_num*conf.batch_size + i].copy(),
                'gt_boxes': val.gt_boxes[batch_num*conf.batch_size + i].copy(),
            }
            #assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)  #as it may contain background also
            no_of_gt_class = len(gt_entry['gt_classes'])   #to get rid of padding
            assert np.max(rels_i) <= no_of_gt_class - 1

            pred_entry = {
                'pred_boxes': (boxes_i * BOX_SCALE / IM_SCALE)[:no_of_gt_class,:],
                'pred_classes': objs_i[:no_of_gt_class],
                'pred_rel_inds': rels_i,
                'gt_rel_inds': gt_rels_ind,
                'obj_scores': obj_scores_i[:no_of_gt_class],
                'rel_scores': pred_scores_i,  # hack for now.
                'obj_preds': obj_preds[:no_of_gt_class],
                'p_o_r_m': p_o_r_m,
                'g_o_r_m': g_o_r_m,
                'pred_edge': pred_edge,
                'true_edge': true_edge,
            }
            assert len(pred_entry['pred_classes'])== len(gt_entry['gt_classes'])

            _, _, _, concat_gt_pred =  evaluator[run_mode].evaluate_scene_graph_entry(gt_entry,pred_entry,)
            if len(evaluator_list)>0:
                eval_entry(run_mode, gt_entry, pred_entry, evaluator, evaluator_list)

            # for a, pr5 in enumerate(concat_gt_pred):
            #     row_count +=1
            #     db.insert_table(values=(a, val.ind_to_classes[pr5[0]], val.ind_to_classes[pr5[1]], val.ind_to_predicates[pr5[2]],
            #                             val.ind_to_predicates[pr5[3]], val.ind_to_predicates[pr5[4]], val.ind_to_predicates[pr5[5]],
            #                             val.ind_to_predicates[pr5[6]], val.ind_to_predicates[pr5[7]]))
        return row_count

    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
    # todo replace wd proper param
    mean_recall = False
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list  = []
    if mean_recall:
        for index, name in enumerate(val_loader.sampler.data_source.ind_to_predicates):
            if index == 0:
                continue
            evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
            if conf.multi_pred:
                evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

    if conf.cache is not None and os.path.exists(conf.cache):
        print("Found {}! Loading from it".format(conf.cache))
        with open(conf.cache,'rb') as f:
            all_pred_entries = pkl.load(f)
        for i, pred_entry in enumerate(tqdm(all_pred_entries)):
            gt_entry = {
                'gt_classes': val.gt_classes[i].copy(),
                'gt_relations': val.relationships[i].copy(),
                'gt_boxes': val.gt_boxes[i].copy(),
            }
            evaluator[run_mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )
        evaluator[run_mode].print_stats()

    else:
        detector.eval()
        for val_b, batch in enumerate(tqdm(val_loader)):
            row_count = val_batch(conf.num_gpus*val_b, batch, evaluator, evaluator_list, row_count)

        evaluator[run_mode].print_stats()
        print("Number of rows : ",row_count)
        #mean_recall = calculate_mR_from_evaluator_list(evaluator_list, run_mode)
        if conf.multi_pred:
            mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode,
                                                                  multiple_preds=conf.multi_pred,
                                                                  save_file='./save_mp_reall.txt')

        if conf.cache is not None:
            with open(conf.cache,'wb') as f:
                pkl.dump(all_pred_entries, f)
    #db.close_conn()

if __name__ == "__main__":
    conf = ModelConfig(file=os.path.join(filepath, 'param.txt'))  # write all param to file
    for mode in ('predcls','sgcls'):
        conf.mode=mode
        eval_rels(conf)
