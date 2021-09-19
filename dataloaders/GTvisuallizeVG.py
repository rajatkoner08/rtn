"""
File that involves dataloaders for the Video VRD dataset.
"""

import json
import os
import pickle
import numpy as np
from dataloaders.visual_genome import VGDataLoader, VG
import cv2
import sys
import torch
from lib.fpn.box_utils import normalized_boxes, union_boxes
from PIL import Image,ImageDraw,ImageFont
from lib.fpn.box_utils import draw_obj_box
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from config import VRD_DATA_DIR,IMG_PATH,OBJ_TID_ROIS,RELATIONS, IM_SCALE
from config import VIDEO_DATA_DIR, FRAME_PATH, VG_SGG_FN, VIDEO_DICT_PATH, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from lib.draw_attention import format_name
from torchvision.transforms import ToPILImage

LIMIT = 99999999
font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 8)

class V_VRD():
    def __init__(self, mode='test', video_path=VIDEO_DATA_DIR, dict_file=VIDEO_DICT_PATH, num_val_im=5000,
                 filter_duplicate_rels=False, filter_non_overlap=True,
                 use_proposals=False):
        """
        Torch dataset loader for Video VRD
        :param mode:
        :param video_path:
        :param num_im:
        :param num_val_im:
        :param filter_duplicate_rels:
        :param filter_non_overlap:
        :param use_proposals:
        :return: filename : [fileID, frameID, imgPath, [tid, type, xmin, ymin, width, height],[s_tid, predicate, o_tid]]
        """
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode

        # Initialize
        self.filter_duplicate_rels = filter_duplicate_rels
        self.video_file = video_path
        self.dict_file_path = dict_file#os.path.join(VIDEO_DICT_PATH, mode)
        self.filter_non_overlap = filter_non_overlap


        #create and save idx file
        if not os.path.isfile(os.path.join(VIDEO_DICT_PATH,'Parsed_Data','obj_class_to_idx.pickle')):
            self.obj_class_to_idx, self.pred_to_idx, self.idx_to_obj_class, self.idx_to_pred = create_indecies(VIDEO_DICT_PATH)
        else:
            # load data if file exists
            with open(os.path.join(dict_file, 'Parsed_Data', 'obj_class_to_idx.pickle'), 'rb') as handle:
                self.obj_class_to_idx = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'predicates_to_idx.pickle'), 'rb') as handle:
                self.pred_to_idx = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'idx_to_predicates.pickle'), 'rb') as handle:
                    self.idx_to_pred = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'idx_to_objects.pickle'), 'rb') as handle:
                self.idx_to_obj_class = pickle.load(handle)
            # with open(os.path.join(dict_file, 'Parsed_Data', 'tid_to_objects.pickle'), 'rb') as handle:
            #     self.tid_to_obj_class = pickle.load(handle)

        #parse json files and load all object and relation
        #filename contains : file no, frame no, img path, object tid with pos, relation with tid, [object class,tid]
        self.filenames = load_video_files(self.mode, self.dict_file_path, self.obj_class_to_idx, self.pred_to_idx)


    ###############################################################################################
def drawImages(datasource, use_obj_comb = False):

        pathname = os.path.join(VRD_DATA_DIR, 'qualitative')
        if not os.path.exists(pathname):
            os.mkdir(pathname)
        print('Saving in dir : ',pathname)
        classname = datasource.ind_to_classes
        predicates = datasource.ind_to_predicates
        for index,file in enumerate(datasource.filenames):
            img_name = os.path.splitext(os.path.basename(file))[0]
            print('Printing image no  : ',index)

            result = datasource.__getitem__(index)
            image_unpadded = result['img']#cv2.imread(file[IMG_PATH])
            img_height, img_width, ratio = result['img_size']
            #todo add obj_comb
            gt_boxes = result['gt_boxes'].copy()*result['scale']
            gt_classes = result['gt_classes'].copy()
            gt_classes_type = format_name(np.asarray(datasource.ind_to_classes)[gt_classes])
            gt_rels = result['gt_relations'].copy()
            obj_comb = result['gt_obj_comb'].copy()
            valid_obj_com = obj_comb[np.where(obj_comb[:,2])]
            unique_rel_obj = np.unique(valid_obj_com[:, :2])
            image_unpadded = ToPILImage()(image_unpadded)
            draw2 = ImageDraw.Draw(image_unpadded)

            def write_rel_in_file(rels):
                if len(rels) == 1:
                    f.write('@{} : {}-{}-{}\n'.format(i+1, gt_classes_type[rels[0][0]], predicates[rels[0][2]],
                                                  gt_classes_type[rels[0][1]]))
                elif len(rels) > 1:
                    for rel in rels:
                        f.write('@{} : {}-{}-{}\n'.format(i+1, classname[gt_classes[rel[0]]], predicates[rel[2]],
                                                      classname[gt_classes[rel[1]]]))

            #draw_obj_box(gt_boxes.copy(), gt_classes_type)
            #draw_obj_box(gt_boxes[unique_rel_obj].copy(), gt_classes_type[unique_rel_obj])

            draw_image(draw2, gt_boxes, gt_classes_type)
            if use_obj_comb:
                # rects = draw_rect(gt_boxes[valid_obj_com[:,0]],gt_boxes[valid_obj_com[:,1]], 7 * 4 - 1)
                u_rois, _ = union_boxes(torch.from_numpy(np.pad(gt_boxes, [(0, 0), (1, 0)], mode='constant')).float(),
                                        torch.from_numpy(
                                            np.pad(valid_obj_com, [(0, 0), (1, 0)], mode='constant')[:, :3]).long())
                u_rois = u_rois[2:3, :]
                draw_image(draw2, u_rois.data.cpu().numpy()[:,1:])
            print('Relation :')
            f = open(os.path.join(pathname,img_name+'_rel.txt'),'w')
            for i, so in enumerate(valid_obj_com):
                fwd_rels = gt_rels[(gt_rels[:, :2] == so[:2]).all(axis=1).nonzero()]
                write_rel_in_file(fwd_rels)
                inv_rels = gt_rels[(gt_rels[:, :2] == so[:2][[1,0]]).all(axis=1).nonzero()]
                write_rel_in_file(inv_rels)

            #cv2.imwrite(os.path.join(pathname, str(index) + '.jpg'),image_unpadded)
            image_unpadded.save(os.path.join(pathname, img_name+ '.jpg'), quality=100, subsampling=0)
            f.close()


def draw_image(draw2, boxes, classes=None):
    color = np.random.randint(255, size=(3, len(boxes), 4))
    for i, gt_box in enumerate(boxes):
        # cv2.rectangle(image_unpadded,(box[0],box[1]),(box[2],box[3]), (255,0,0), 2)
        if classes is None:
            draw2 = draw_box(draw2, gt_box, cls_ind=0, text_str=str(i), color = tuple(color[0,i,...]))
        else:
            draw2 = draw_box(draw2, gt_box, cls_ind=0, text_str=classes[i], color = tuple(color[1,i,...]))

def draw_box(draw, boxx, cls_ind, text_str, color):
        box = tuple([float(b) for b in boxx])

        # color = tuple([int(x) for x in cmap(cls_ind)])
        width = 2
        # draw the fucking box
        draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=width)
        draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=width)
        draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=width)
        draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=width)

        # draw.rectangle(box, outline=color)
        w, h = draw.textsize(text_str, font=font)

        x1text = box[0]
        y1text = max(box[1] - h, 0)
        x2text = min(x1text + w, draw.im.size[0])
        y2text = y1text + h
        # print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        #     h, w, x1text, y1text, x2text, y2text))

        draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
        draw.text((x1text, y1text), text_str, fill='black', font=font)
        return draw

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def minmax(value):
    return min(max(value,0),1)

def draw_rect(boxes1, boxes2, pooling_size=27):
    box_pairs = np.concatenate((boxes1, boxes2),1)
    N = box_pairs.shape[0]
    uboxes = np.zeros((N,2,pooling_size, pooling_size))

    for n in range(N):
        x1_union = min(box_pairs[n, 0], box_pairs[n, 4])
        y1_union = min(box_pairs[n, 1], box_pairs[n, 5])
        x2_union = max(box_pairs[n, 2], box_pairs[n, 6])
        y2_union = max(box_pairs[n, 3], box_pairs[n, 7])

        w = x2_union - x1_union
        h = y2_union - y1_union

        for i in range(2):
            # Now everything is in the range [0, pooling_size].
            x1_box = (box_pairs[n, 0 + 4 * i] - x1_union) * pooling_size / w
            y1_box = (box_pairs[n, 1 + 4 * i] - y1_union) * pooling_size / h
            x2_box = (box_pairs[n, 2 + 4 * i] - x1_union) * pooling_size / w
            y2_box = (box_pairs[n, 3 + 4 * i] - y1_union) * pooling_size / h
            # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x1_box, y1_box, x2_box, y2_box))
            for j in range(pooling_size):
                y_contrib = minmax(j + 1 - y1_box) * minmax(y2_box - j)
                for k in range(pooling_size):
                    x_contrib = minmax(k + 1 - x1_box) * minmax(x2_box - k)
                    # print("j {} yc {} k {} xc {}".format(j, y_contrib, k, x_contrib))
                    uboxes[n, i, j, k] = x_contrib * y_contrib
    return uboxes


def getObjType(objects, tid, obj_to_idx):
    """

    :param objects:
    :param tid:
    :param obj_to_idx:
    :return:
    """
    for d in objects:
        if d['tid']==tid:
            return obj_to_idx[d['category']]


def create_indecies(dict_file):
    """

    :param dict_file:
    :return:
    """
    object_count = 0
    predicate_count = 0
    global_tid_count = 0
    objects_to_idx = {}
    predicates_to_idx = {}
    idx_to_objects = {}
    idx_to_predicates = {}
    #tid_to_object = {}

    allDirs = ['train','test', 'val']
    for dir in allDirs:
        files = os.listdir(os.path.join(dict_file,dir))
        for file in files:
            info = json.load(open(os.path.join(dict_file,dir,file),'r'))
            objects = info['subject/objects']
            predicates = info['relation_instances']

            #load object id, tid
            for o in objects:
                obj_category = o['category']
                if not obj_category in idx_to_objects.values():
                    idx_to_objects[object_count] = obj_category
                    objects_to_idx[obj_category] = object_count
                    object_count +=1
                # tid_to_object[global_tid_count] = objects_to_idx[obj_category]
                # global_tid_count +=1

            #load relation
            for r in predicates:
                relation = r['predicate']
                if not relation in idx_to_predicates.values():
                    idx_to_predicates[predicate_count] = relation
                    predicates_to_idx[relation] = predicate_count
                    predicate_count +=1

    #save data after completion of parsing to save time
    with open(os.path.join(dict_file,'Parsed_Data','obj_class_to_idx.pickle'), 'wb') as handle:
        pickle.dump(objects_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dict_file,'Parsed_Data','idx_to_objects.pickle'), 'wb') as handle:
        pickle.dump(idx_to_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dict_file,'Parsed_Data','predicates_to_idx.pickle'), 'wb') as handle:
        pickle.dump(predicates_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dict_file,'Parsed_Data','idx_to_predicates.pickle'), 'wb') as handle:
        pickle.dump(idx_to_predicates, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return objects_to_idx,predicates_to_idx, idx_to_objects, idx_to_predicates

if __name__=="__main__":
    train, val, test = VG.splits(num_val_im=5000,filter_duplicate_rels=True,
                          use_proposals=False, filter_non_overlap=False, require_overlap = True)
    val = test
    train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=1,
                                                   num_workers=0,
                                                   num_gpus=1,vg_mini =True)
    drawImages(val_loader.dataset, use_obj_comb =False)