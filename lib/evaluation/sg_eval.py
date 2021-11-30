"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
import math
import pickle
from functools import reduce
from lib.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import MODES,Rel_Threshold,EDGE_THRESHOLD

np.set_printoptions(precision=3)

class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False, topfive_pred=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: [], 'class_accuracy': []}
        self.attr_acc=[]
        self.rel_pred_acc = []
        self.edge_pred_acc = []
        self.edge_recall = []
        self.rel_recall = []
        self.multiple_preds = multiple_preds
        self.topfive_pred = topfive_pred

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict, self.rel_pred_acc,
                                 self.edge_pred_acc, self.edge_recall, self.rel_recall, self.attr_acc, topfive=self.topfive_pred, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds)
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self, epoch_num=None, writer=None, return_output=False):
        #if not return_output:
        print('======================' + self.mode + '============================')
        output = {}
        for k, v in self.result_dict[self.mode + '_recall'].items():
            if k == 'class_accuracy':
                if not return_output:
                    print('Class Accuracy: ',np.mean(v))
                if writer is not None:
                    writer.add_scalar('class accuracy', np.mean(v), epoch_num)
            else:
                #if not return_output:
                print('R@%i: %f' % (k, np.mean(v)))
                output['R@%i' % k] = np.mean(v)
                if writer is not None:
                    writer.add_scalar('data/R@%i'%(k), np.mean(v), epoch_num)
        if return_output:
            return  output
        #now write the relation prediction accuracy
        if len(self.rel_pred_acc)>0 :
            print('Relation prediction Accuracy :',np.mean(self.rel_pred_acc))
            print('Predicted/ Actual Relation :', np.mean(self.rel_recall))
            if writer is not None:
                writer.add_scalar('pred_rel_acc', np.mean(self.rel_pred_acc), epoch_num)
                writer.add_scalar('pred_vs_actual_rel', np.mean(self.rel_recall), epoch_num)
        # now write the correct edge prediction accuracy
        if len(self.edge_pred_acc) > 0:
            print('Edge prediction Accuracy :', np.mean(self.edge_pred_acc))
            #print('Predicted/ Actual Edge :', np.mean(self.edge_recall))
            if writer is not None:
                writer.add_scalar('edge_pred_acc', np.mean(self.edge_pred_acc), epoch_num)
                #writer.add_scalar('pred_vs_actual_edge', np.mean(self.edge_recall), epoch_num)
        #now write the attribute prediction accuracy
        if len(self.attr_acc) >0 :
            print('Attribute accuracy : ',np.mean(self.attr_acc))
            if writer is not None:
                writer.add_scalar('attr_pred_acc', np.mean(self.attr_acc), epoch_num)

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, rel_pred_acc, edge_pred_acc,
                       edge_recall, rel_recall, attr_acc, multiple_preds=False, topfive=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict: 
    :param viz_dict: 
    :param kwargs: 
    :return: 
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']
    gt_attrs = gt_entry['gt_attrs']


    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    pred_attrs = pred_entry['pred_attrs']

    pred_classes_temp =  pred_entry['obj_preds'] #to avoid class_accu =1 for predcls
    #now  how accurate the relation proposal network is
    if pred_entry['p_o_r_m'] is not None:
        g_o_r_m = pred_entry['g_o_r_m']
        p_o_r_m = pred_entry['p_o_r_m']
        p_o_r_m[p_o_r_m< Rel_Threshold] = 0.01
        p_o_r_m[p_o_r_m >= Rel_Threshold] = 1

    if pred_entry['true_edge'] is not None:
        true_edge = pred_entry['true_edge']
        pred_edge = np.where(pred_entry['pred_edge']>EDGE_THRESHOLD, 1, 0.1)  #0.1 is used that its not equal to 0

    if mode == 'predcls':
        pred_boxes = gt_boxes
        pred_classes = gt_classes       #todo for sorting print change here
        obj_scores = np.ones(gt_classes.shape[0])
    elif mode == 'sgcls':
        pred_boxes = gt_boxes
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'sgdet' or mode == 'phrdet':
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    else:
        raise ValueError('invalid mode')

    if multiple_preds:
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:,0]], score_inds[:,1]+1))
        predicate_scores = rel_scores[score_inds[:,0], score_inds[:,1]+1]
    else:
        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        predicate_scores = rel_scores[:,1:].max(1)

    if mode in ('predcls', 'sgcls') and topfive:
        gt_rels_ind = pred_entry['gt_rel_inds']
        concat_gt_pred = np.column_stack((pred_classes[gt_rels_ind[:,:2]],gt_rels_ind[:,2],
                                          1+np.argsort(rel_scores[:, 1:], axis=1)[:, ::-1][:,:5])) #[s,o,r0,r1,r2,r3,r4]

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
                mode, gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)

    if mode in ('predcls', 'sgcls', 'sgdet', 'phrdet'):
        class_accuracy = (gt_classes==pred_classes_temp).sum()/len(gt_classes)
        if pred_entry['p_o_r_m'] is not None:
            rel_pred_acc.append((p_o_r_m==g_o_r_m).sum()/g_o_r_m.sum())
            rel_recall.append(p_o_r_m.sum()/g_o_r_m.sum())
        if pred_entry['pred_edge'] is not None:
            edge_pred_acc.append((pred_edge==true_edge).sum()/true_edge.sum())
            edge_recall.append(np.count_nonzero(pred_edge==1)  / true_edge.sum())
        if pred_attrs is not None:
            ## for threshold of 0.5 accuracy should be 100% while from accepted answer you will get 66.67% so
            pred_attrs[pred_attrs >= 0.5] = 1
            pred_attrs[pred_attrs < 0.5] = 0  ## assign 0 label to those with less than 0.5
            if len(gt_attrs)>0:
                attr_acc.append(len(np.argwhere((gt_attrs !=0) & (pred_attrs !=0))) / len(gt_attrs))

    for k in result_dict[mode + '_recall']:
        if k == 'class_accuracy':
            result_dict[mode + '_recall'][k].append(class_accuracy)
        else:
            match = reduce(np.union1d, pred_to_gt[:k])    #todo check for inreasing match
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)

    return pred_to_gt, pred_5ples, rel_scores, concat_gt_pred if topfive else None

    # print(" ".join(["R@{:2d}: {:.3f}".format(k, v[-1]) for k, v in result_dict[mode + '_recall'].items()]))
    # Deal with visualization later
    # # Optionally, log things to a separate dictionary
    # if viz_dict is not None:
    #     # Caution: pred scores has changed (we took off the 0 class)
    #     gt_rels_scores = pred_scores[
    #         gt_rels[:, 0],
    #         gt_rels[:, 1],
    #         gt_rels[:, 2] - 1,
    #     ]
    #     # gt_rels_scores_cls = gt_rels_scores * pred_class_scores[
    #     #         gt_rels[:, 0]] * pred_class_scores[gt_rels[:, 1]]
    #
    #     viz_dict[mode + '_pred_rels'] = pred_5ples.tolist()
    #     viz_dict[mode + '_pred_rels_scores'] = max_pred_scores.tolist()
    #     viz_dict[mode + '_pred_rels_scores_cls'] = max_rel_scores.tolist()
    #     viz_dict[mode + '_gt_rels_scores'] = gt_rels_scores.tolist()
    #     viz_dict[mode + '_gt_rels_scores_cls'] = gt_rels_scores_cls.tolist()
    #
    #     # Serialize pred2gt matching as a list of lists, where each sublist is of the form
    #     # pred_ind, gt_ind1, gt_ind2, ....
    #     viz_dict[mode + '_pred2gt_rel'] = pred_to_gt


###########################
def evaluate_recall(mode, gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    #aggregate all d confidence score
    scores_overall = relation_scores.prod(1)
    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5) and not mode == 'predcls':
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def calculate_mR_from_evaluator_list(evaluator_list, mode, multiple_preds=False, save_file=None):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        # print('\n')
        # print('relationship: ', pred_name)
        rel_results = evaluator_rel[mode].print_stats(return_output=True)
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']):
            continue
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall


    if multiple_preds:
        recall_mode = 'mean recall without constraint'
    else:
        recall_mode = 'mean recall with constraint'
    print('\n')
    print('======================' + mode + '  ' + recall_mode + '============================')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)

    if save_file is not None:
        if multiple_preds:
            save_file = save_file.replace('.pkl', '_multiple_preds.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(all_rel_results, f)

    return mean_recall


def eval_entry(mode, gt_entry, pred_entry, evaluator, evaluator_list):
    evaluator[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    for (pred_id, _, evaluator_rel) in evaluator_list:
        gt_entry_rel = gt_entry.copy()
        mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
        if gt_entry_rel['gt_relations'].shape[0] == 0:
            continue

        evaluator_rel[mode].evaluate_scene_graph_entry(
            gt_entry_rel,
            pred_entry,
        )