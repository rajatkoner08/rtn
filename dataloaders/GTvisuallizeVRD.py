"""
File that involves dataloaders for the Video VRD dataset.
"""

import json
import os
import pickle
import numpy as np
import torch
import cv2
import sys
from pathlib import Path
import numbers
from PIL import Image,ImageDraw,ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from config import VRD_DATA_DIR,IMG_PATH,OBJ_TID_ROIS,RELATIONS, OBJ_CLASS
from config import VIDEO_DATA_DIR, FRAME_PATH, VG_SGG_FN, VIDEO_DICT_PATH, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from util.mot_util import rescaleBBox,scale_bbox
from torchvision.transforms import ToPILImage

LIMIT = 99999999


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
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)

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
    def __drawImages__(self):
        pathname = os.path.join(VRD_DATA_DIR, 'qualitative')
        if not os.path.exists(pathname):
            os.mkdir(pathname)
        for index,file in enumerate(self.filenames):
            print('Printing image no  : ',index)
            tid_to_obj_type = {}
            image_unpadded = Image.open(file[IMG_PATH]).convert('RGB')#cv2.imread(file[IMG_PATH])
            img_height = image_unpadded.height
            img_width = image_unpadded.width
            tform = [
                SquarePad(),
                Resize(IM_SCALE),
                ToTensor()
            ]
            transform_pipeline = Compose(tform)
            image_tensor = transform_pipeline(image_unpadded)
            gt_boxes = file[OBJ_TID_ROIS].copy()
            gt_rels = file[RELATIONS].copy()
            obj_class_tid = file[OBJ_CLASS]
            image_unpadded = ToPILImage()(image_tensor.squeeze())
            draw2 = ImageDraw.Draw(image_unpadded)
            for i, gt_box in enumerate(gt_boxes):
                obj_type = self.idx_to_obj_class[obj_class_tid[i][0]]
                obj_tid = obj_class_tid[i][1]
                box = gt_box[1::]
                #box = np.asarray([box[0],box[1],box[0]+box[2],box[1]+box[3]])
                rescaleBox = rescaleBBox(box,img_width,img_height)
                tid_to_obj_type[obj_tid] = obj_type
                #cv2.rectangle(image_unpadded,(box[0],box[1]),(box[2],box[3]), (255,0,0), 2)
                draw2 = self.__draw_box__(draw2, rescaleBox,
                                 cls_ind=0,
                                 text_str=obj_type + '_' + str(obj_tid))
            #cv2.imwrite(os.path.join(pathname, str(index) + '.jpg'),image_unpadded)
            image_unpadded.save(os.path.join(pathname, str(index) + '.jpg'), quality=100, subsampling=0)


    def __draw_box__(self, draw, boxx, cls_ind, text_str):
        box = tuple([float(b) for b in boxx])
        if '-GT' in text_str:
            color = (255, 128, 0, 255)
        else:
            color = (0, 128, 0, 255)

        # color = tuple([int(x) for x in cmap(cls_ind)])

        # draw the fucking box
        draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
        draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
        draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
        draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)

        # draw.rectangle(box, outline=color)
        w, h = draw.textsize(text_str, font=self.font)

        x1text = box[0]
        y1text = max(box[1] - h, 0)
        x2text = min(x1text + w, draw.im.size[0])
        y2text = y1text + h
        # print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        #     h, w, x1text, y1text, x2text, y2text))

        draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
        draw.text((x1text, y1text), text_str, fill='black', font=self.font)
        return draw

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_video_files(mode, dict_file, obj_to_idx, pred_to_idx):
    """

    :param mode:
    :param dict_file:
    :param obj_to_idx:
    :param pred_to_idx:
    :param tid_to_obj_class:
    :return:
    """

    all_info = []
    # for dir_idx, mode in enumerate(allDirs):
    allFiles = os.listdir(os.path.join(dict_file,mode))

    for file_idx, file in enumerate(allFiles):
        accumulated_info = []
        relations = []
        info = json.load(open(os.path.join(dict_file,mode,file), 'r'))
        trajectories = info['trajectories']
        objects = info['subject/objects']
        predicates = info['relation_instances']

        #loop over all frame where trajectory exists
        for i_frame,trj in enumerate(trajectories):
            # todo remove these line
            # if file == 'ILSVRC2015_train_00418000.json' and i_frame == 705:
            #     print("Test")
            if len(trj)>0: #skip blank frames
                obj_id_pos = []
                obj_classes = []
                for k, obj in enumerate(trj):
                    tid = obj['tid']
                    obj_type = getObjType(objects, tid,obj_to_idx)
                    xmin = obj['bbox']['xmin']
                    ymin = obj['bbox']['ymin']
                    x_max = obj['bbox']['xmax']
                    y_max = obj['bbox']['ymax']
                    obj_id_pos.append([tid, xmin , ymin, x_max, y_max])
                    obj_classes.append([obj_type,tid])
                #add to main list
                accumulated_info.append([i_frame, np.asanyarray(obj_id_pos), np.asanyarray(obj_classes)])
                # if file_idx ==1 and j == 30:
                #     print('Test')

        #loop over all relation and push it frame data
        for k, pred in enumerate(predicates):
            for l  in range(pred['begin_fid'], pred['end_fid']):
                #use this one as per frame id
                relations.append([l, pred['subject_tid'], pred['object_tid'], pred_to_idx[pred['predicate']]])
        #covert to array for indexing
        relations = np.asanyarray(relations)

        for frame in accumulated_info:
            img_path = os.path.join(FRAME_PATH, os.path.splitext(file)[0], '%04d.jpg'%frame[0])
            relations_in_frame = relations[relations[:,0]==frame[0]][:,1::]
            #todo remove line
            # if len(relations_in_frame)==0:#todo if u dnt have relation for trajectory dn ignore it
            #     print(frame[0],' and file ', file)
            if len(relations_in_frame)>0:
                #saving format : file no, frame no, img path, object tid with pos, relation with tid, [object class,tid]
                all_info.append([file_idx, frame[0], img_path, frame[1], relations_in_frame, frame[2]])


    return all_info

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

def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob



if __name__=="__main__":
    test_data = V_VRD()
    test_data.__drawImages__()