# create predictions from the other stuff
"""
Go from proposals + scores to relationships.

pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression

in all cases we'll return:
boxes, objs, rels, pred_scores

"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time

def filter_dets(batch_size, boxes, obj_max_scores, obj_classes, rel_inds, pred_scores, pred_obj_rel_mat,
                gt_obj_rel_mat, obj_dist, edge_dist, gt_edge, attr_dist=None, inference=False):
    """
    Filters detections....
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_max_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    out = []
    pred_scores_max, pred_classes_argmax = pred_scores.data[:, 1:].max(1)

    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    obj_preds = np.argmax(obj_dist.data.cpu().numpy(),1)
    obj_scores_all_np = obj_max_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_np = boxes.data.cpu().numpy()
    rel_inds_np =  rel_inds.data.cpu().numpy()
    pred_scores_max = pred_scores_max.data.cpu().numpy()
    pred_scores_np = pred_scores.data.cpu().numpy()
    if pred_obj_rel_mat is not None:
        pred_obj_rel_mat_np = pred_obj_rel_mat.data.cpu().numpy()
        gt_obj_rel_mat_np = gt_obj_rel_mat.data.cpu().numpy()

    if attr_dist is not None:
        attr_dist_np = attr_dist.data.cpu().numpy()

    if edge_dist is not None:
        pred_edge = edge_dist.view(batch_size, edge_dist.shape[0] // batch_size).data.cpu().numpy()
        true_edge = gt_edge.view(batch_size, gt_edge.shape[0] // batch_size).data.cpu().numpy()

    obj_preds = obj_preds.reshape(batch_size, obj_classes.shape[0] // batch_size)
    objs_np = objs_np.reshape(batch_size, obj_classes.shape[0] // batch_size)
    max_obj_scores_np = obj_scores_all_np.reshape(batch_size, obj_max_scores.shape[0] // batch_size)
    obj_dist_np = obj_dist.data.cpu().numpy().reshape(batch_size, obj_max_scores.shape[0] // batch_size, -1)
    boxes_np = boxes_np.reshape(batch_size, boxes.shape[0] // batch_size, 4)

    image_ofset = objs_np.shape[1]

    p_o_r_m = None
    g_o_r_m = None
    pred_edge_batch = None
    gt_edge_batch = None
    for i, (batch_box, max_batch_obj_scores, batch_obj_dist, batch_obj_cls, batch_obj_preds) in enumerate(zip(boxes_np, max_obj_scores_np, obj_dist_np, objs_np, obj_preds)):

        batch_rels_ind = np.where(rel_inds_np[:,0]==i)[0]
        rels_ind = rel_inds_np[batch_rels_ind][:,2:]
        batch_pred_scores_max = pred_scores_max[batch_rels_ind]
        batch_pred_scores = pred_scores_np[batch_rels_ind]

        if pred_obj_rel_mat is not None:
            assert pred_obj_rel_mat_np.shape[2] == pred_obj_rel_mat_np.shape[1] == gt_obj_rel_mat_np.shape[1] == gt_obj_rel_mat_np.shape[2] == len(batch_obj_cls)
            p_o_r_m = pred_obj_rel_mat_np[i, :, :]
            g_o_r_m = gt_obj_rel_mat_np[i, :, :]

        if edge_dist is not None: #todo insert a assert statement
            pred_edge_batch = pred_edge[i,:]
            gt_edge_batch = true_edge[i,:]

        if attr_dist is not None:
            batch_attr_dist = attr_dist_np[np.where(attr_dist_np[:,0]==i)[0],1:]
        else:
            batch_attr_dist = None

        num_box = batch_box.shape[0]
        assert max_batch_obj_scores.shape[0] == num_box

        assert  len(batch_obj_cls) == len(max_batch_obj_scores) == len(batch_obj_preds)
        num_rel = rels_ind.shape[0]

        assert batch_pred_scores_max.shape[0] == num_rel

        obj_scores0 = obj_scores_all_np[rels_ind[:,0]]
        obj_scores1 = obj_scores_all_np[rels_ind[:,1]]

        rels_ind[:,:2] -= i*image_ofset

        rel_scores_argmaxed = batch_pred_scores_max * obj_scores0 * obj_scores1
        rel_scores_idx = np.argsort(rel_scores_argmaxed.reshape(-1), axis=0)[::-1]

        if rels_ind.shape[1] == 2:
            gt_rels_ind = None
            rels_ind = rels_ind[rel_scores_idx]
        elif rels_ind.shape[1] == 3:
            gt_rels_ind = rels_ind[rel_scores_idx]
            rels_ind = rels_ind[:,:2][rel_scores_idx]


        out.append((batch_box, batch_obj_cls, batch_obj_dist if inference else max_batch_obj_scores, batch_attr_dist, rels_ind, gt_rels_ind,
                    batch_pred_scores[rel_scores_idx], batch_obj_preds, p_o_r_m, g_o_r_m, pred_edge_batch, gt_edge_batch))

    return out
    #return boxes_np, objs_np, obj_scores_np, np.asarray(rels_out), np.asarray(pred_scores_out)

# def _get_similar_boxes(boxes, obj_classes_topk, nms_thresh=0.3):
#     """
#     Assuming bg is NOT A LABEL.
#     :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
#     :param obj_classes: [num_box, topk] class labels
#     :return: num_box, topk, num_box, topk array containing similarities.
#     """
#     topk = obj_classes_topk.size(1)
#     num_box = boxes.size(0)
#
#     box_flat = boxes.view(-1, 4) if boxes.dim() == 3 else boxes[:, None].expand(
#         num_box, topk, 4).contiguous().view(-1, 4)
#     jax = bbox_overlaps(box_flat, box_flat).data > nms_thresh
#     # Filter out things that are not gonna compete.
#     classes_eq = obj_classes_topk.data.view(-1)[:, None] == obj_classes_topk.data.view(-1)[None, :]
#     jax &= classes_eq
#     boxes_are_similar = jax.view(num_box, topk, num_box, topk)
#     return boxes_are_similar.cpu().numpy().astype(np.bool)
