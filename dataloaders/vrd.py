import h5py
import numpy as np
import os
import pickle
import json
from torch.utils.data import Dataset,DataLoader
from dataloaders.blob import Blob
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, LABEL_DATA_DIR, BOX_SCALE, IM_SCALE, PROPOSAL_FN,BG_EDGE_PER_IMG, FG_EDGE_PER_IMG
from collections import defaultdict
from dataloaders.visual_genome import get_obj_comb_and_rels,get_obj_rels_mat,get_normalized_boxes, assertion_checks,vg_collate

class VRD(Dataset):
    '''
    Visual Translation Embedding Network for Visual Relation Detection
    Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua
    '''
    def __init__(self, vrd_path, mode='train'):

        sg_json = os.path.join(vrd_path, 'json_dataset/annotations_' + mode + '.graph')
        self.ind_to_classes = json.load(open(os.path.join(vrd_path, 'json_dataset/objects.graph'),'r'))
        self.ind_to_predicates = json.load(open(os.path.join(vrd_path, 'json_dataset/predicates.graph'),'r'))

        with open(sg_json, 'r') as read_file:
            self.graphs = json.load(read_file)
        self.mode = mode
        assert self.mode in ['train','test'], self.mode
        self.corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']

        self.mode_ = ('test' if self.mode == 'test' else 'train')
        img_path = os.path.join(vrd_path,'sg_dataset')

        self.gt_boxes = []
        self.gt_classes = []
        self.relationships = []
        self.filenames = []
        test_box = []
        for image_id, (img_key,img_data) in enumerate(self.graphs.items()):
            # Borrowed from https://github.com/yangxuntu/vrd/blob/6c2e3f36129ea506f263efa34f95abd3e88a819c/tf-faster-rcnn-master/tools/vg_process_dete.py
            sub_boxes = []
            sub_classes = []
            obj_boxes = []
            obj_classes = []
            predicates = []
            img_name = os.path.join(img_path, img_key)
            for i, pred_data in enumerate(img_data):
                sub_boxes.append(pred_data['subject']['bbox'])
                obj_boxes.append(pred_data['object']['bbox'])
                sub_classes.append(pred_data['subject']['category'])
                obj_classes.append(pred_data['object']['category'])
                predicates.append(pred_data['predicate'])

            if len(predicates)>0:
                gt_boxes, unique_inds, boxes_inds = np.unique(np.concatenate((sub_boxes, obj_boxes), axis=0),
                                                              axis=0, return_index=True, return_inverse=True)
                gt_classes = np.concatenate((sub_classes, obj_classes), axis=0)[unique_inds]

                n = len(boxes_inds) // 2
                relationships = np.column_stack((boxes_inds[:n], boxes_inds[n:], np.asarray(predicates)[:, None]+1))  # +1 because the background will be added

                self.gt_classes.append(gt_classes)
                self.gt_boxes.append(gt_boxes)
                self.relationships.append(relationships)
                self.filenames.append(img_name)
                test_box = np.concatenate((test_box, gt_boxes))  if len(test_box)>0 else gt_boxes

            else:
                print('test')
        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, test

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
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)

        unique_comb, fwd_rels, inv_rels = get_obj_comb_and_rels(gt_rels, self.gt_classes[index].copy(), gt_boxes,
                                                                self.use_bg_rels, require_overlap=self.require_overlap)
        obj_rel_mat = get_obj_rels_mat(gt_rels, self.gt_classes[index])
        gt_norm_boxes = get_normalized_boxes(gt_boxes.copy(), image_unpadded)  # todo change as per im scale
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
            'obj_rel_mat': obj_rel_mat,
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



class VRDDataLoader(DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=0, num_gpus=3, mode='det', vg_mini = False, o_val=False,
               **kwargs):
        assert mode in ('det', 'rel')
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

if __name__ == "__main__":
    vrd = VRD('/nfs/data/koner/data/VRD')