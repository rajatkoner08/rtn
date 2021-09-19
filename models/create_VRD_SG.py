"""
Visualization script. I used this to create the figures in the paper.

WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.video_vrd import VRDDataLoader, V_VRD
from lib.context_model import TransformerModel
import numpy as np
import torch
from lib.evaluation.sg_eval import _triplet
import pickle
from config import ModelConfig,DETECTOR_CONF,REL_CONF
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE,DATA_PATH,VIDEO_DICT_PATH
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from functools import reduce
from util.mot_util import draw_box,text_wrap
from torchvision.transforms import ToPILImage

conf = ModelConfig()
train = V_VRD(mode='train')
test = V_VRD(mode='test')#todo for testing its changed to val
train_loader, test_loader = VRDDataLoader.splits(train, test, mode='rel',batch_size=2, num_workers=1, num_gpus=1)

#load all the dict file for indexing
with open(os.path.join(VIDEO_DICT_PATH, 'Parsed_Data', 'VG_ind_to_classes.pickle'), 'rb') as handle:
    VG_ind_to_classes = pickle.load(handle)
with open(os.path.join(VIDEO_DICT_PATH, 'Parsed_Data', 'VG_ind_to_predicates.pickle'), 'rb') as handle:
    VG_ind_to_predicates = pickle.load(handle)
with open(os.path.join(VIDEO_DICT_PATH, 'Parsed_Data', 'idx_to_objects.pickle'), 'rb') as handle:
    idx_to_obj_class = pickle.load(handle)
with open(os.path.join(VIDEO_DICT_PATH, 'Parsed_Data', 'idx_to_predicates.pickle'), 'rb') as handle:
    idx_to_pred = pickle.load(handle)

detector = TransformerModel(classes=VG_ind_to_classes, rel_classes=VG_ind_to_predicates,
                            num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                            use_resnet=conf.use_resnet, order=conf.order,
                            nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                            use_proposals=conf.use_proposals,
                            pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                            pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                            pooling_dim=conf.pooling_dim,
                            rec_dropout=conf.rec_dropout,
                            use_bias=conf.use_bias,
                            use_tanh=conf.use_tanh,
                            limit_vision=conf.limit_vision
                            )
# torch.cuda.set_device(1)
# torch.cuda.get_device_capability(device=torch.cuda.current_device())
detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])


############################################ HELPER FUNCTIONS ###################################

def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index)) * (255 - pad) + pad)

    return map_index_to_rgb_color


cmap = get_cmap(len(VG_ind_to_classes) + 1)

def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(tqdm(test_loader)):
        if val_b%100 == 0:
            val_batch(conf.num_gpus * val_b, batch, val_b)
            print('Saved image number ',val_b)

    evaluator[conf.mode].print_stats()


def val_batch(batch_num, b,itr_no, method='', thrs=(20, 50, 100)):
    det_res = detector[b]
    # if conf.num_gpus == 1:
    #     det_res = [det_res]
    assert conf.num_gpus == 1
    boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res
    #todo classes and relation need to change as per vg, gt_relation classes are wrong
    gt_entry = {
        'gt_classes': b.gt_classes.data.cpu().numpy()[:,1],
        'gt_relations': b.gt_rels.data.cpu().numpy()[:,1::],
        'gt_boxes': b.gt_boxes.data.cpu().numpy(),
        'gt_tids' : b.gt_tids.data.cpu().numpy()[:,1],
    }
    assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

    #full prediction result from model
    pred_entry = {
        'pred_boxes': boxes_i,
        'pred_classes': objs_i,
        'pred_rel_inds': rels_i,
        'obj_scores': obj_scores_i,
        'rel_scores': pred_scores_i,
    }
    pred_triplet = []
    gt_triplet = []
    pred_filtred = []
    # todo use a dict for class id to tid, and modify this code
    for i, rel in enumerate(gt_entry['gt_relations']):
        # convert from tid to subject/object id
        sub = gt_entry['gt_classes'][np.where(gt_entry['gt_tids'] == rel[0])][0]
        obj = gt_entry['gt_classes'][np.where(gt_entry['gt_tids'] == rel[1])][0]
        gt_triplet.append([idx_to_obj_class[sub], idx_to_pred[rel[2]], idx_to_obj_class[obj]])

    #todo : remove testing block
    if method == 'confidence':
        # Get a list of objects with good confidence from prediction
        objs_match = (pred_entry['obj_scores']>=DETECTOR_CONF)
        pred_boxes = pred_entry['pred_boxes'][objs_match]
        pred_classes = (pred_entry['pred_classes'][objs_match])
        #get predicates which invlolves previously predicted object class
        #pred_match = np.multiply.reduce((np.isin(rels_i, pred_classes)), axis=-1)
        #todo replace with nicer code
        pred_match = (np.isin(objs_i[rels_i], pred_classes))
        pred_matching_relation = []
        for pred in pred_match:
            if pred[0]==True and pred[1]==True:
                pred_matching_relation.append(True)
            else:
                pred_matching_relation.append(False)
        pred_matching_relation = np.asanyarray(pred_matching_relation)
        pred_sub_obj = pred_entry['pred_rel_inds'][pred_matching_relation]
        predicates = pred_entry['rel_scores'][pred_matching_relation].argmax(1)
        predicate_score = pred_entry['rel_scores'][:, 1:].max(1)
        sub_obj_classes = pred_entry['pred_classes'][pred_sub_obj]

        #write sub-relation-obj for prediction and ground truth
        for i, rel in enumerate(pred_sub_obj):
            if predicate_score[pred_matching_relation][i]>0.2:
                pred_triplet.append([VG_ind_to_classes[sub_obj_classes[i][0]],VG_ind_to_predicates[predicates[i]],VG_ind_to_classes[sub_obj_classes[i][1]]])
                if rel[0] not in pred_filtred:
                    pred_filtred.append(rel[0])
                if rel[1] not in pred_filtred:
                    pred_filtred.append(rel[1])
        if len(pred_filtred)>0:
            pred_filtred = np.asanyarray(pred_filtred)
            pred_boxes = pred_entry['pred_boxes'][pred_filtred]
            pred_classes = pred_entry['pred_classes'][pred_filtred]
    else:
        # show only those results which have 50% overlap with GT Objects
        # Get a list of objects that match, and GT objects that dont
        objs_match = (bbox_overlaps(pred_entry['pred_boxes'], gt_entry['gt_boxes']) >= 0.5)
        objs_matched = objs_match.any(1)
        pred_classes = pred_entry['pred_classes'][objs_matched]
        pred_boxes = pred_entry['pred_boxes'][objs_matched].astype(float)
        matched_cls_scores = pred_entry['obj_scores'][objs_matched]
        if len(pred_classes) > 1:
            searchedSet = set(np.where(objs_matched)[0])
            #searchedSet  = set([1,2,59,9])
            matched_pred_ind = [searchedSet.issuperset(row) for row in pred_entry['pred_rel_inds'].tolist()]
            matched_relation = pred_entry['pred_rel_inds'][matched_pred_ind]
            matched_pred = 1+pred_entry['rel_scores'][:, 1:].argmax(1)[matched_pred_ind]
            rel_scores = pred_entry['rel_scores'][:, 1:].max(1)[matched_pred_ind]

            # todo remove Dirty hack display
            pred_classes = pred_entry['pred_classes'][list(set(matched_relation[np.where(rel_scores > REL_CONF)].flatten()))]
            pred_boxes = pred_entry['pred_boxes'][list(set(matched_relation[np.where(rel_scores > REL_CONF)].flatten()))]
            # write sub-relation-obj for prediction and ground truth
            for i, rel in enumerate(matched_relation):
                if rel_scores[i]>=REL_CONF :
                    pred_triplet.append([VG_ind_to_classes[pred_entry['pred_classes'][rel[0]]], VG_ind_to_predicates[matched_pred[i]],
                                             VG_ind_to_classes[pred_entry['pred_classes'][rel[1]]]])
        else:
            pred_triplet.append('No match Found')

    #open and resize image and draw relation and boxes, then save it
    resize_img = Image.new('RGB', (IM_SCALE, IM_SCALE), (255, 255, 255))
    orig_img = Image.open(test[itr_no]['img_path']).convert('RGB')
    orig_img.thumbnail((IM_SCALE, IM_SCALE), Image.ANTIALIAS)
    resize_img.paste(orig_img,(0,0))
    gt_draw = ImageDraw.Draw(resize_img)
    resize_img2 = resize_img.copy()
    pred_draw = ImageDraw.Draw(resize_img2)
    #write relation at buttom of image
    gt_draw = text_wrap(gt_draw, text_list=gt_triplet)
    pred_draw = text_wrap(pred_draw, text_list=pred_triplet)
    if itr_no == 360:
        print('Test')
    for i, box in enumerate(gt_entry['gt_boxes']):
        gt_draw = draw_box(gt_draw, box,
                         cls_ind=gt_entry['gt_classes'][i],
                         text_str=idx_to_obj_class[gt_entry['gt_classes'][i]])

    for i, box in enumerate(pred_boxes):
        pred_draw = draw_box(pred_draw, box,
                         cls_ind=pred_classes[i],
                         text_str=VG_ind_to_classes[pred_classes[i]])

    #now save those boxes and relation
    pathname = os.path.join(DATA_PATH, 'qualitative')
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    resize_img.save(os.path.join(pathname, 'img'+str(itr_no)+'.jpg'), quality=100, subsampling=0)
    resize_img2.save(os.path.join(pathname, 'imgbox'+str(itr_no)+'.jpg'), quality=100, subsampling=0)


mAp = val_epoch()
