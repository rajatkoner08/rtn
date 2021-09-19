"""
File that involves dataloaders for the Video VRD dataset.
"""

import json
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from config import VRD_DATA_DIR,IMG_PATH,OBJ_TID_ROIS,RELATIONS, OBJ_CLASS
from config import VIDEO_DATA_DIR, FRAME_PATH, VG_SGG_FN, VIDEO_DICT_PATH, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from collections import defaultdict
from util.mot_util import rescaleBBox


class V_VRD(Dataset):
    def __init__(self, mode, video_path=VIDEO_DATA_DIR, dict_file=VIDEO_DICT_PATH, num_val_im=5000,
                 filter_duplicate_rels=False, filter_non_overlap=True,use_pickle=True, seq_pad= True,
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
        self.dict_file_path = dict_file
        self.filter_non_overlap = filter_non_overlap
        self.seq_pad = seq_pad

        if use_pickle : #saves time
            with open(os.path.join(dict_file,'Parsed_Data', mode+'_file.pickle'), 'rb') as handle:
                self.filenames =pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'obj_class_to_idx.pickle'), 'rb') as handle:
                self.obj_class_to_idx = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'predicates_to_idx.pickle'), 'rb') as handle:
                self.pred_to_idx = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'idx_to_predicates.pickle'), 'rb') as handle:
                    self.idx_to_pred = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', 'idx_to_objects.pickle'), 'rb') as handle:
                self.idx_to_obj_class = pickle.load(handle)
            with open(os.path.join(dict_file, 'Parsed_Data', mode+'_file.pickle'), 'rb') as handle:
                self.filenames = pickle.load(handle)

        else:   #create and parse file
            self.obj_class_to_idx, self.pred_to_idx, self.idx_to_obj_class, self.idx_to_pred = create_indecies(
                self.dict_file_path)
            # filename contains : file no, frame no, img path, object tid with pos, relation with tid, [object class,tid]
            self.filenames = load_video_files(self.mode, self.dict_file_path, self.obj_class_to_idx, self.pred_to_idx)

        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    ###############################################################################################
    def __getitem__(self, index):
        image_unpadded = Image.open(self.filenames[index][IMG_PATH]).convert('RGB')
        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        gt_boxes = self.filenames[index][OBJ_TID_ROIS].copy()
        scaled_boxes, im_size = rescaleBBox( gt_boxes[:,1::].astype(np.float32), image_unpadded.width, image_unpadded.height)
        gt_rels = self.filenames[index][RELATIONS].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)
        # if self.seq_pad:
        #     lenToPad = np.zeros((self.maxNumOfObj - len(scaled_boxes),4),dtype=float)
        #     scaled_boxes = np.concatenate((scaled_boxes, lenToPad), axis=0)
        from itertools import chain
        if self.is_train:
            all_unique_rel= []
            for rel in gt_rels[:,0:2]:
                if  np.sort(rel).tolist() not in  all_unique_rel:
                    all_unique_rel.append(np.sort(rel).tolist())
            all_unique_rel = np.asarray(all_unique_rel)
           # rearrange relation as per object combination order
            fwd_rels = None
            inv_rels = None
            for i, each_comb in enumerate(all_unique_rel):
                fwd_indices = np.where((gt_rels[:,:2]==each_comb).all(axis=1))[0]
                if len(fwd_indices)>0:
                    if fwd_rels is not None:
                        fwd_temp = np.column_stack((np.full((len(fwd_indices)),i), gt_rels[fwd_indices][:,2]))
                        fwd_rels = np.concatenate([fwd_rels, fwd_temp], axis=0)  #np.concatenate([fwd_rels, gt_rels[fwd_indices]], axis=0)
                    else:
                        fwd_rels = np.column_stack((np.full((len(fwd_indices)),i), gt_rels[fwd_indices][:,2])) #gt_rels[fwd_indices]
                inv_indices = np.where((gt_rels[:,:2]==each_comb[[1,0]]).all(axis=1))[0]
                if len(inv_indices)>0:
                    if inv_rels is not None:
                        inv_temp = np.column_stack((np.full((len(inv_indices)),i), gt_rels[inv_indices][:,2]))
                        inv_rels = np.concatenate([inv_rels, inv_temp], axis=0) #np.concatenate([inv_rels, gt_rels[inv_indices]], axis=0)
                    else:
                        inv_rels =  np.column_stack((np.full((len(inv_indices)),i), gt_rels[inv_indices][:,2])) #gt_rels[inv_indices]

        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': scaled_boxes,
            'gt_classes': self.filenames[index][OBJ_CLASS][:,0].copy(),
            'gt_tids' : self.filenames[index][OBJ_CLASS][:,1].copy(),
            'gt_relations': gt_rels,
            'fwd_relations' : fwd_rels,
            'inv_relations' : inv_rels,
            'gt_obj_comb' : all_unique_rel,
            'img_path': self.filenames[index][IMG_PATH],
            'flipped': flipped
        }
        # if self.rpn_rois is not None:
        #     entry['proposals'] = self.rpn_rois[index]

        return entry

    def __len__(self):
        #number of datasamples
         return len(self.filenames)

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @property
    def num_predicates(self):
        return len(self.pred_to_idx)

    @property
    def num_classes(self):
        return len(self.obj_class_to_idx)

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
    for file_idx, file in enumerate(tqdm(allFiles)):
        accumulated_info = []
        relations = []
        info = json.load(open(os.path.join(dict_file,mode,file), 'r'))
        trajectories = info['trajectories']
        objects = info['subject/objects']
        predicates = info['relation_instances']

        #loop over all frame where trajectory exists
        for i_frame,trj in enumerate(trajectories):
           # if i_frame == 30:
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
                    obj_id_pos.append([tid, xmin, ymin, x_max, y_max])
                    obj_classes.append([obj_type,tid])
                #add to main list
                accumulated_info.append([i_frame, np.asanyarray(obj_id_pos), np.asanyarray(obj_classes)])

        #loop over all relation and push it frame data
        for k, pred in enumerate(predicates):
            for l  in range(pred['begin_fid'], pred['end_fid']):
                # if pred['end_fid']==45:
                #     print("test")
                #check if relation already exists in the list
                if(checkUnique(np.asarray(relations),pred['subject_tid'],  pred['object_tid'], pred_to_idx[pred['predicate']], l)):
                    #use this one as per frame id
                    relations.append([l, pred['subject_tid'], pred['object_tid'], pred_to_idx[pred['predicate']]])
                    #relations.append([l, search(objects, pred['subject_tid']), search(objects, pred['object_tid']), pred_to_idx[pred['predicate']]])
        #covert to array for indexing
        relations = np.asanyarray(relations)
        i = 0
        for frame in accumulated_info:
            # if i==30:
            #     print("Test at video_vrs.py")
            i += 1
            img_path = os.path.join(FRAME_PATH, os.path.splitext(file)[0], '%04d.jpg'%frame[0])
            relations_in_frame = relations[relations[:,0]==frame[0]][:,1::]
            if len(relations_in_frame)>0:
                # if relations_in_frame.shape[0] != np.unique(relations_in_frame, axis=0).shape[0]:
                #relations_in_frame = np.unique(relations_in_frame, axis=0)
                relation_with_obj_pos = obj_position(relations_in_frame,frame[2])
                #saving format : file no, frame no, img path, object tid with pos, relation with tid, [object class,tid]
                all_info.append([file_idx, frame[0], img_path, frame[1], relation_with_obj_pos, frame[2]])
        file_idx+=1
    #save into pickle
    with open(os.path.join(dict_file, 'Parsed_Data', mode + '_file.pickle'), 'wb') as handle:
        pickle.dump(self.filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

def obj_position(relations_in_frame,obj_tid):
    updated_relation_with_pos = []

    def searchObj_pos(tid):
        return [i for i, element in enumerate(obj_tid) if element[1] == tid][0]
    for realtion in relations_in_frame:
        updated_relation_with_pos.append([searchObj_pos(realtion[0]), searchObj_pos(realtion[1]), realtion[2]])

    return  np.asarray(updated_relation_with_pos)


def checkUnique(relations, sub, obj, predicate, frame_no):
    if len(relations)>0:
        relations_in_frame = relations[relations[:, 0] == frame_no][:,1::]
        test = any(np.array_equal(x,(sub,obj,predicate)) for x in relations_in_frame)
        return not test
    else:
        return True

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

def vg_collate(data, pad_batch, num_gpus=3, is_train=False, mode='det'):
    """
    #todo no padding for gt_realtion, do as per vrd progress
    :param data:
    :param num_gpus:
    :param is_train:
    :param mode:
    :param pad_batch: pad till max number of obj with every sample for gt_boxes, gt_classes
    :return:
    """
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    if pad_batch:
        maxNumOfObjInBatch = max(len(frame['gt_boxes']) for frame in data)
        maxNumOfObjCombInBatch = max(len(frame['gt_obj_comb']) for frame in data)
    for d in data:
        #todo this will not work for SGDET, dn fix number of obj is needed
        if pad_batch:
            numOfObj = len(d['gt_boxes'])
            numOfObjComb = len(d['gt_obj_comb'])
            d['gt_boxes'] = np.concatenate((d['gt_boxes'],np.zeros((( maxNumOfObjInBatch - numOfObj ),4),dtype=float)), axis=0)
            d['gt_classes'] = np.concatenate((d['gt_classes']+1, np.zeros((maxNumOfObjInBatch - numOfObj ), dtype=int)), axis=0)
            d['gt_tids'] = np.concatenate((d['gt_tids'] + 1, np.zeros((maxNumOfObjInBatch - numOfObj ), dtype=int)), axis=0)
            d['src_seq'] = np.concatenate((np.ones((numOfObj),dtype=int), np.zeros((maxNumOfObjInBatch - numOfObj),dtype=int)))

            d['gt_obj_comb'] = np.concatenate((d['gt_obj_comb'], np.zeros(((maxNumOfObjCombInBatch - numOfObjComb), 2), dtype=int)))
            d['tgt_seq'] = np.concatenate((np.ones((numOfObjComb), dtype=int), np.zeros((maxNumOfObjCombInBatch - numOfObjComb), dtype=int)))
        blob.append(d)
    blob.reduce()
    return blob

class VRDDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, pad_batch, batch_size=1, num_workers=2, num_gpus=1, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True, pad_batch=pad_batch),
            drop_last=True,
            # pin_memory=True,
            **kwargs
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False, pad_batch=pad_batch),
            drop_last=True,
            # pin_memory=True,
            **kwargs
        )
        return train_load, val_load

def set_random_seed():
    seed = np.random.randint(0, 100)
    print('Random Seed : ',seed)
    np.random.seed(seed)

if __name__=="__main__":
    train_data = V_VRD(mode='train', filter_duplicate_rels=False, num_val_im=5000)