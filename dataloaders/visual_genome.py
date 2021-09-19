"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os
import random
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
#from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.fpn.box_utils import bbox_overlaps
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, LABEL_DATA_DIR, BOX_SCALE, IM_SCALE, PROPOSAL_FN,BG_EDGE_PER_IMG, FG_EDGE_PER_IMG
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from collections import defaultdict
from lib.CustomSampler import CustomSampler
from lib.helper import get_fwd_inv_rels


class VG(Dataset):
    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=LABEL_DATA_DIR,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, require_overlap = False,o_valid =False,
                 use_proposals=False,use_bg_rels =True, use_bg =True):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode
        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.require_overlap = require_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'
        self.use_bg_rels = use_bg_rels
        self.use_bg = use_bg    #todo use bg and rels are probably duplicate, plz check

        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels, orig_split=o_valid,
            filter_non_overlap=self.filter_non_overlap and self.is_train,
        )

        self.filenames = load_image_filenames(image_file)
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)

        if use_proposals:
            print("Loading proposals", flush=True)
            p_h5 = h5py.File(PROPOSAL_FN, 'r')
            rpn_rois = p_h5['rpn_rois']
            rpn_scores = p_h5['rpn_scores']
            rpn_im_to_roi_idx = np.array(p_h5['im_to_roi_idx'][self.split_mask])
            rpn_num_rois = np.array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np.column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]))

        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = None # COCO()    #todo coco is removed becz of downgrade
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    def __getitem__(self, index):
        image_unpadded = Image.open(self.filenames[index]).convert('RGB')

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        gt_boxes = self.gt_boxes[index].copy()

        # Boxes are already at BOX_SCALE
        if self.is_train:
            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1])
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0])

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        w, h = image_unpadded.size
        box_scale_factor = BOX_SCALE / max(w, h)

        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max(w, h)
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        gt_rels = self.relationships[index].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)

        unique_comb, fwd_rels, inv_rels = get_obj_comb_and_rels(gt_rels,self.gt_classes[index].copy(), gt_boxes, self.use_bg_rels, require_overlap=self.require_overlap)
        obj_rel_mat = get_obj_rels_mat(gt_rels, self.gt_classes[index])
        gt_norm_boxes = get_normalized_boxes(gt_boxes.copy(), image_unpadded)   #todo change as per im scale
        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': gt_rels,
            'fwd_relations': fwd_rels,
            'inv_relations': inv_rels,
            'gt_obj_comb': unique_comb,
            'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
            'index': index,
            'flipped': flipped,
            'fn': self.filenames[index],
            'norm_boxes': gt_norm_boxes,
            'obj_rel_mat' : obj_rel_mat,
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_obj_comb_and_rels(rels, gt_classes, gt_boxes, use_bg_rels, require_overlap):
    '''
    It create all posiible combination of rels which have some intersction or they have rels in gt
    :param rels:
    :param gt_classes:
    :param gt_boxes:
    :param use_bg_rels:
    :return:
    '''
    #create mix of bg and fg rels in sorted order unique does that
    unique_fg_edge = np.unique(np.sort(rels[:, :2], axis=1),axis=0)
    if use_bg_rels:
        all_comb =  np.arange(len(gt_classes))[:, None] != np.arange(len(gt_classes))[None]
        if require_overlap:
            all_comb = np.stack(np.nonzero(np.triu(all_comb & (bbox_overlaps(gt_boxes, gt_boxes) > 0))), axis=-1)
            #all_comb = np.concatenate((all_comb,rels[:, 0:2]))
            #all_comb = np.concatenate((all_comb, unique_fg_edge))
            #unique_bg_edge = np.unique(all_comb, axis=0)
            #all_unique_rel = unique_bg_edge
        else:
            all_comb = np.stack(np.nonzero(np.triu(all_comb)), axis=-1)
    else:
        all_comb = unique_fg_edge
        assert 'This condition is not yet implemented'

    #now filter out the fg edge
    unique_bg_edge = []
    for edge in all_comb:
        if not (edge == unique_fg_edge).all(axis=1).any():
            unique_bg_edge.append(edge)
    #check if the number of rels exceeds the limit, then sample
    if len(unique_fg_edge)>FG_EDGE_PER_IMG:
        unique_fg_edge = np.asarray(random.sample(list(unique_fg_edge),k=FG_EDGE_PER_IMG))
    if len(unique_bg_edge)>BG_EDGE_PER_IMG and use_bg_rels:
        unique_bg_edge = np.asarray(random.sample(list(unique_bg_edge),k=BG_EDGE_PER_IMG))

    if len(unique_bg_edge)>0:
        all_comb = np.unique(np.concatenate((np.asarray(unique_bg_edge), unique_fg_edge)), axis=0)
    else:
        all_comb = unique_fg_edge

    all_comb = np.append(all_comb, np.zeros(all_comb.shape[0], dtype=np.int64)[:, None], axis=1)
    #all_comb = all_comb[np.argsort(all_comb[:,0]*(len(gt_classes)**2)+all_comb[:,0]*len(gt_classes)+all_comb[:,1])]

    # # return all unique object combinations and fwd, inv relations
    # all_unique_rel = []     #todo replace with helper functions
    # for rel in all_comb:
    #     if np.sort(rel).tolist() not in all_unique_rel:
    #         all_unique_rel.append(np.sort(rel).tolist())
    # all_unique_rel = np.asarray(all_unique_rel)

    # def stack_and_cat(to_rels, indices, obj1, obj2):
    #     """
    #     concatinate with previous relation if they exists, else make relation for background
    #     """
    #     if len(indices) > 0:
    #         if to_rels is not None:
    #             rel_temp = np.column_stack((np.full((len(indices)), i), rels[indices]))
    #             to_rels = np.concatenate([to_rels, rel_temp], axis=0)
    #         else:
    #             to_rels = np.column_stack((np.full((len(indices)), i), rels[indices]))
    #     else:
    #         if to_rels is not None:
    #             rel_temp = np.column_stack(( i, obj1, obj2, 0))
    #             to_rels = np.concatenate([to_rels, rel_temp], axis=0)
    #         else:
    #             to_rels = np.column_stack(( i, obj1, obj2, 0))
    #
    #     return to_rels
    #     # rearrange relation as per object combination order, format [pos of unq comb, rel number]
    #
    # #split to fwd and inv relation and add bg relation
    # fwd_rels = None
    # inv_rels = None
    # for i, each_comb in enumerate(all_comb):
    #     fwd_indices = np.where((rels[:, :2] == each_comb[:2]).all(axis=1))[0]
    #     fwd_rels = stack_and_cat(fwd_rels, fwd_indices, each_comb[0], each_comb[1])
    #
    #     inv_indices = np.where((rels[:, :2] == each_comb[:2][[1, 0]]).all(axis=1))[0]
    #     inv_rels = stack_and_cat(inv_rels, inv_indices, each_comb[1], each_comb[0])
    #
    #     if len(fwd_indices) >0 or len(inv_indices)>0:
    #         each_comb[2]=1

    return get_fwd_inv_rels(all_comb, rels)

def get_obj_rels_mat(gt_rels, gt_classes):
    obj_rel_mat = np.full((len(gt_classes), len(gt_classes)), 0)
    for rel in gt_rels:
        obj_rel_mat[rel[0],rel[1]] = 1
        obj_rel_mat[rel[1], rel[0]] = 1
    return  obj_rel_mat


def get_normalized_boxes(norm_boxes, image_unpadded):
    unscaled_img = np.array([BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0],
                             BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1]])
    norm_boxes = norm_boxes.astype(float)
    norm_boxes[:, 0] /= unscaled_img[0]
    norm_boxes[:, 1] /= unscaled_img[1]
    norm_boxes[:, 2] /= unscaled_img[0]
    norm_boxes[:, 3] /= unscaled_img[1]

    return norm_boxes


def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
    #assert len(fns) == 108073
    return fns


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False, orig_split=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    if orig_split:
        split = 2 if mode == 'test' else 0
    else:
        split = 2 if mode in ['test', 'val'] else 0
    #split = 2 if mode in ['test', 'val'] else 0  #only when u gonna use validation from test
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[num_val_im:]
    if num_val_im > 0:
        if mode == 'val':           #take few samples from test as val
            image_index = image_index[:num_val_im]
        elif mode == 'train' and orig_split:                       #todo use this flag to switch between original and modified val set
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)

    #add padding in batches
    maxNumOfObjInBatch = max(len(frame['gt_boxes']) for frame in data)
    maxNumOfObjCombInBatch = max(len(frame['gt_obj_comb']) for frame in data)

    for d in data:
        #todo this will not work for SGDET, dn fix number of obj is needed
        numOfObj = len(d['gt_boxes'])
        numOfObjComb = len(d['gt_obj_comb'])
        obj2rel = np.full((maxNumOfObjInBatch, maxNumOfObjInBatch), 0)

        d['gt_boxes'] = np.concatenate((d['gt_boxes'],np.zeros((( maxNumOfObjInBatch - numOfObj ),4),dtype=float)), axis=0)
        d['norm_boxes'] = np.concatenate((d['norm_boxes'], np.zeros(((maxNumOfObjInBatch - numOfObj), 4), dtype=float)), axis=0)
        d['gt_classes'] = np.concatenate((d['gt_classes'], np.zeros((maxNumOfObjInBatch - numOfObj ), dtype=int)), axis=0)
        d['src_seq'] = np.concatenate((np.ones((numOfObj),dtype=int), np.zeros((maxNumOfObjInBatch - numOfObj),dtype=int)))
        d['gt_obj_comb'] = np.concatenate((d['gt_obj_comb'], np.zeros(((maxNumOfObjCombInBatch - numOfObjComb), 3), dtype=int)))
        d['tgt_seq'] = np.concatenate((np.ones((numOfObjComb), dtype=int), np.zeros((maxNumOfObjCombInBatch - numOfObjComb), dtype=int)))
        d['obj_comb_pos'] = numOfObjComb
        obj2rel[:numOfObj,:numOfObj] += d['obj_rel_mat']
        d['obj_rel_mat'] = obj2rel

        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=0, num_gpus=3, mode='det', vg_mini = False, o_val=False,
               **kwargs):
        assert mode in ('det', 'rel')
        sampler = CustomSampler(train_data, vg_mini, o_val)
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            #sampler = sampler,
            #worker_init_fn = torch.set_rng_state(torch.initial_seed()),
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size, #* num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load
