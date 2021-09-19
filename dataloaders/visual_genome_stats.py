"""
Staticial scripts of objects and their relations
"""

from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import os
from PIL import ImageDraw
from lib.helper import draw_box, get_unique_rels, missed_rels, load_unscaled,get_decoded_rels,inrease_rect
filepath = os.path.dirname(os.path.abspath(__file__))
from config import ModelConfig, VRD_DATA_DIR
from lib.fpn.box_utils import bbox_overlaps
from dataloaders.sql_file import Sql_ops

conf = ModelConfig(file = os.path.join(filepath,'param.txt')) #write all param to file
if conf.model == 'motifnet':
    pass

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

# val = test  #todo unlock to perform on Test

_, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)
class_name = val_loader.dataset.ind_to_classes
rels_name = val_loader.dataset.ind_to_predicates

def total_rels(rels):
    total_count = 0
    for rel in rels:
        total_count+=len(rel)
    return  total_count

def count_interset(val_rel,val_boxes):
    for i, rels in enumerate(val_rel):
        intrsect = bbox_overlaps(val_boxes[i], val_boxes[i])
        positions = np.transpose(np.nonzero(intrsect))

def draw_imgs_rels(dataset, index, boxes=None, classes=None):
    resecaled_image =load_unscaled(dataset.filenames[index]).copy()
    draw = ImageDraw.Draw(resecaled_image)
    if boxes is None and classes is None:
        boxes = dataset.gt_boxes[index].copy()
        classes = dataset.gt_classes[index].copy()

    for i, (box, obj_class) in enumerate(zip(boxes, classes)):
          draw_box(draw, box[[0,1,2, 3]], None, class_name[obj_class]+'_'+str(i))
    pathname = os.path.join(VRD_DATA_DIR, 'qualitative')
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    resecaled_image.save(os.path.join(pathname, str(index) + '.jpg'), quality=100, subsampling=0)
    print('Saved image number {}'.format(index))

def determine_rels(all_bboxes, gt_rels,gt_classes, require_overlap, dist_thresold = 1.7, draw_img=False):
    all_rels = 0
    all_missed_rels = 0
    #computer rels based on IOU, distance, area
    for i, (bboxes, rels, classes) in enumerate(zip( all_bboxes,gt_rels, gt_classes )):
        n_obj = np.zeros((len(bboxes)))
        possible_obj_comb = n_obj[:, None] == n_obj[None]
        np.fill_diagonal(possible_obj_comb, 0)

        #if dist_thresold > 1:
        large_bboxes = inrease_rect(bboxes.copy(),dist_thresold)
        # Require overlap for detection
        if require_overlap and len(n_obj) >= 6:  # todo hacking noww
            possible_obj_comb = possible_obj_comb & (bbox_overlaps(large_bboxes,
                                                                   large_bboxes) >0 )
        possible_obj_comb = np.triu(possible_obj_comb)
        computed_rels = np.argwhere(possible_obj_comb==1)

        all_rels += len(rels)
        unique_rels_dict = get_unique_rels(rels[:,:2])
        unique_rels = np.asarray(list(unique_rels_dict.keys()))

        #get the rels which is not captured
        missed = missed_rels(unique_rels, computed_rels)
        all_missed_rels += sum([unique_rels_dict[val] for val in missed])

        if draw_img and len(missed)>0:
            draw_imgs_rels(val_loader.dataset, i, bboxes[np.unique(list(missed))], classes[np.unique(list(missed))])
            print(get_decoded_rels(rels, classes, np.asarray(list(missed)), class_name, rels_name))
    print('Total percentage of missed rels : ',(all_missed_rels/all_rels)*100)

def write_rels(gt_classes, val_rel):
    #write all rels to database
    db = Sql_ops()
    rels_folder = os.path.join(VRD_DATA_DIR, 'rels')
    db.create_table()
    db.del_table()
    if not os.path.exists(rels_folder):
        os.makedirs(rels_folder)
    for i, (rels, classes) in enumerate(zip(val_rel, gt_classes)):
        for rel in rels:
            db.insert_table(values=(class_name[classes[rel[0]]], rels_name[rel[2]],class_name[classes[rel[1]]]))
    #db.close_conn()

###################################################################
############## All the function calling############################

print('Starting stas.....')
val_boxes = val_loader.dataset.gt_boxes
val_rel = val_loader.dataset.relationships
val_classes = val_loader.dataset.gt_classes

# train_boxes = train_loader.dataset.gt_boxes
# train_rel = train_loader.dataset.relationships
# train_classes = train_loader.dataset.gt_classes

##################################################

write_rels(val_classes, val_rel)
# determine_rels(val_boxes, val_rel, val_classes, require_overlap=True)
# total_rels =  total_rels(val_rel)
# total_intersected_rels = count_interset(val_rel,val_boxes)
#draw_imgs_rels(val_loader.dataset, index=2608)

print('Ending stats....')
