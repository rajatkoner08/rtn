"""
Data blob, hopefully to make collating less painful and MGPU training possible
"""
from lib.fpn.anchor_targets import anchor_target_layer
from lib.fpn.box_utils import normalized_boxes
import numpy as np
import torch
from torch.autograd import Variable


class Blob(object):
    def __init__(self, mode='det', is_train=False, num_gpus=1, primary_gpu=0, batch_size_per_gpu=3, pad_batch=True, dataset='vg'):
        """
        Initializes an empty Blob object.
        :param mode: 'det' for detection and 'rel' for det+relationship
        :param is_train: True if it's training
        """
        assert mode in ('det', 'rel')
        assert num_gpus >= 1
        self.mode = mode
        self.is_train = is_train
        self.num_gpus = num_gpus
        self.batch_size_per_gpu = batch_size_per_gpu
        self.primary_gpu = primary_gpu
        self.pad_batch = pad_batch
        self.dataset = dataset

        self.imgs = []  # [num_images, 3, IM_SCALE, IM_SCALE] array
        self.im_sizes = []  # [num_images, 4] array of (h, w, scale, num_valid_anchors)
        self.all_anchor_inds = []  # [all_anchors, 2] array of (img_ind, anchor_idx). Only has valid
        # boxes (meaning some are gonna get cut out)
        self.all_anchors = []  # [num_im, IM_SCALE/4, IM_SCALE/4, num_anchors, 4] shapes. Anchors outside get squashed
                               # to 0
        self.gt_boxes = []  # [num_gt, 4] boxes
        self.gt_classes = []  # [num_gt,2] array of img_ind, class
        self.gt_rels = []  # [num_rels, 3]. Each row is (gtbox0, gtbox1, rel).
        self.fwd_rels = []  # [num_rels, 2]. Each row is (obj fwd comb, rel)
        self.inv_rels = []  # [num_rels, 2]. Each row is (obj inv comb, rel)
        self.src_seq = []  # valid num of obj in a frame
        self.tgt_seq = []  # valid num of obj combination in a frame
        self.gt_obj_comb = [] #all unique object combination
        self.obj_comb_pos = []  # all unique object combination pos
        self.norm_boxes = []  #  [num_gt, 4] boxes that are normalized
        self.obj_rel_mat = []
        self.gt_attr = []

        self.gt_sents = []
        self.gt_nodes = []
        self.sent_lengths = []

        self.train_anchor_labels = []  # [train_anchors, 5] array of (img_ind, h, w, A, labels)
        self.train_anchors = []  # [train_anchors, 8] shapes with anchor, target

        self.train_anchor_inds = None  # This will be split into GPUs, just (img_ind, h, w, A).

        self.batch_size = None
        self.gt_box_chunks = None
        self.gt_attr_chunks = None
        self.anchor_chunks = None
        self.train_chunks = None
        self.obj_rel_mat_chunks = None
        self.proposal_chunks = None
        self.gt_obj_comb_chunks = None
        self.obj_comb_pos_chunks = None
        self.u_img_chunks = None
        self.proposals = []

    @property
    def is_flickr(self):
        return self.mode == 'flickr'

    @property
    def is_rel(self):
        return self.mode == 'rel'

    @property
    def volatile(self):
        return not self.is_train

    def append(self, d):
        """
        Adds a single image to the blob
        :param datom:
        :return:
        """
        i = len(self.imgs)
        self.imgs.append(d['img'])

        h, w, scale = d['img_size']

        # all anchors
        self.im_sizes.append((h, w, scale))

        gt_boxes_ = d['gt_boxes'].astype(np.float32)* d['scale']  #todo imp -reduces to 592 scale
        self.gt_boxes.append(np.column_stack((i * np.ones(d['gt_boxes'].shape[0], dtype=np.int64),gt_boxes_)))

        self.norm_boxes.append(np.column_stack((i * np.ones(d['gt_boxes'].shape[0], dtype=np.int64),normalized_boxes(gt_boxes_.copy(), w, h).astype(np.float32))))   #todo move to ctx_model

        self.gt_classes.append(np.column_stack((
            i * np.ones(d['gt_classes'].shape[0], dtype=np.int64),
            d['gt_classes'],
        )))

        if self.dataset =='gqa':
            self.gt_attr.append(np.column_stack((
                i * np.ones(d['gt_attr'].shape[0], dtype=np.int64),
                d['gt_attr'],
            )))
        self.obj_rel_mat.append(d['obj_rel_mat'])

        #add padding for valid sequences
        if self.pad_batch:
            self.src_seq.append(np.column_stack((
            i * np.ones(d['src_seq'].shape[0], dtype=np.int64),
            d['src_seq'])))

            self.tgt_seq.append(np.column_stack((
                i * np.ones(d['tgt_seq'].shape[0], dtype=np.int64),
                d['tgt_seq'])))


        # Add relationship info
        if self.is_rel:
            self.gt_rels.append(np.column_stack((
                i * np.ones(d['gt_relations'].shape[0], dtype=np.int64),
                d['gt_relations'])))

            self.fwd_rels.append(np.column_stack((
                    i * np.ones(d['fwd_relations'].shape[0], dtype=np.int64),
                    d['fwd_relations'])))

            if d['inv_relations'] is not None:
                self.inv_rels.append(np.column_stack((
                    i * np.ones(d['inv_relations'].shape[0], dtype=np.int64),
                    d['inv_relations'])))

            self.gt_obj_comb.append(np.column_stack((
                i * np.ones(d['gt_obj_comb'].shape[0], dtype=np.int64),
                d['gt_obj_comb'])))



            self.obj_comb_pos.append(d['obj_comb_pos'])

        # Augment with anchor targets
        if self.is_train:
            train_anchors_, train_anchor_inds_, train_anchor_targets_, train_anchor_labels_ = \
                anchor_target_layer(gt_boxes_, (h, w))

            self.train_anchors.append(np.hstack((train_anchors_, train_anchor_targets_)))

            self.train_anchor_labels.append(np.column_stack((
                i * np.ones(train_anchor_inds_.shape[0], dtype=np.int64),
                train_anchor_inds_,
                train_anchor_labels_,
            )))

        if 'proposals' in d:
            self.proposals.append(np.column_stack((i * np.ones(d['proposals'].shape[0], dtype=np.float32),
                                                   d['scale'] * d['proposals'].astype(np.float32))))



    def _chunkize(self, datom, tensor=torch.LongTensor):
        """
        Turn data list into chunks, one per GPU
        :param datom: List of lists of numpy arrays that will be concatenated.
        :return:
        """
        chunk_sizes = [0] * self.num_gpus
        for i in range(self.num_gpus):
            for j in range(self.batch_size_per_gpu):
                chunk_sizes[i] += datom[i * self.batch_size_per_gpu + j].shape[0]
        return Variable(tensor(np.concatenate(datom, 0)), volatile=self.volatile), chunk_sizes

    def reduce(self):
        """ Merges all the detections into flat lists + numbers of how many are in each"""
        if len(self.imgs) != self.batch_size_per_gpu * self.num_gpus:
            raise ValueError("Wrong batch size? imgs len {} bsize/gpu {} numgpus {}".format(
                len(self.imgs), self.batch_size_per_gpu, self.num_gpus
            ))

        self.imgs = Variable(torch.stack(self.imgs, 0), volatile=self.volatile)
        self.obj_rel_mat = torch.FloatTensor(np.stack(self.obj_rel_mat,0)) # todo use it as like img
        self.im_sizes = np.stack(self.im_sizes).reshape((self.num_gpus, self.batch_size_per_gpu, 3))

        if self.dataset=='gqa':
            self.gt_attr, self.gt_attr_chunks = self._chunkize(self.gt_attr)

        if self.is_rel:
            self.gt_rels, self.gt_rel_chunks = self._chunkize(self.gt_rels)
            self.fwd_rels, self.fwd_rels_chunks = self._chunkize(self.fwd_rels)
            self.inv_rels, self.inv_rels_chunks = self._chunkize(self.inv_rels)

        self.gt_boxes, self.gt_box_chunks = self._chunkize(self.gt_boxes, tensor=torch.FloatTensor)
        #self.obj_rel_mat, self.obj_rel_mat_chunks = self._chunkize(self.obj_rel_mat)
        self.norm_boxes, _ = self._chunkize(self.norm_boxes, tensor=torch.FloatTensor)
        self.gt_classes, _ = self._chunkize(self.gt_classes)
        self.src_seq, _ = self._chunkize(self.src_seq)
        self.tgt_seq, _ = self._chunkize(self.tgt_seq)
        self.gt_obj_comb, self.gt_obj_comb_chunks = self._chunkize(self.gt_obj_comb)
        self.obj_comb_pos, self.obj_comb_pos_chunks = torch.tensor(np.transpose(self.obj_comb_pos)), self.num_gpus * [self.batch_size_per_gpu]

        if self.is_train:
            self.train_anchor_labels, self.train_chunks = self._chunkize(self.train_anchor_labels)
            self.train_anchors, _ = self._chunkize(self.train_anchors, tensor=torch.FloatTensor)
            self.train_anchor_inds = self.train_anchor_labels[:, :-1].contiguous()

        if len(self.proposals) != 0:
            self.proposals, self.proposal_chunks = self._chunkize(self.proposals, tensor=torch.FloatTensor)



    def _scatter(self, x, chunk_sizes, dim=0):
        """ Helper function"""
        if self.num_gpus == 1:
            return x.cuda(self.primary_gpu, non_blocking=True)
        return torch.nn.parallel.scatter_gather.Scatter.apply(
            list(range(self.num_gpus)), chunk_sizes, dim, x)

    def scatter(self):
        """ Assigns everything to the GPUs"""
        self.imgs = self._scatter(self.imgs, [self.batch_size_per_gpu] * self.num_gpus)
        self.obj_rel_mat = self._scatter(self.obj_rel_mat, [self.batch_size_per_gpu] * self.num_gpus)

        self.gt_classes_primary = self.gt_classes.cuda(self.primary_gpu, non_blocking=True)
        self.gt_boxes_primary = self.gt_boxes.cuda(self.primary_gpu, non_blocking=True)

        # Predcls might need these
        if self.dataset=='gqa':
            self.gt_attr = self._scatter(self.gt_attr, self.gt_attr_chunks)
        self.gt_classes = self._scatter(self.gt_classes, self.gt_box_chunks)
        self.gt_boxes = self._scatter(self.gt_boxes, self.gt_box_chunks)
        self.norm_boxes = self._scatter(self.norm_boxes, self.gt_box_chunks)
        self.src_seq = self._scatter(self.src_seq, self.gt_box_chunks)
        self.tgt_seq = self._scatter(self.tgt_seq, self.gt_box_chunks)
        #self.obj_rel_mat = self._scatter(self.obj_rel_mat, self.obj_rel_mat_chunks)

        if self.is_train:

            self.gt_obj_comb = self._scatter(self.gt_obj_comb, self.gt_obj_comb_chunks)
            self.obj_comb_pos = self._scatter(self.obj_comb_pos, self.obj_comb_pos_chunks)
            self.train_anchor_inds = self._scatter(self.train_anchor_inds,
                                                   self.train_chunks)
            self.train_anchor_labels = self.train_anchor_labels.cuda(self.primary_gpu, non_blocking=True)
            self.train_anchors = self.train_anchors.cuda(self.primary_gpu, non_blocking=True)

            if self.is_rel:
                self.gt_rels = self._scatter(self.gt_rels, self.gt_rel_chunks)
                self.fwd_rels = self._scatter(self.fwd_rels, self.fwd_rels_chunks)
                self.inv_rels = self._scatter(self.inv_rels, self.inv_rels_chunks)

        else:
            self.gt_obj_comb = self.gt_obj_comb.cuda(self.primary_gpu)
            self.obj_comb_pos = self.obj_comb_pos.cuda(self.primary_gpu)
            if self.is_rel:
                self.gt_rels = self.gt_rels.cuda(self.primary_gpu, non_blocking=True)
                self.fwd_rels = self.fwd_rels.cuda(self.primary_gpu)
                self.inv_rels = self.inv_rels.cuda(self.primary_gpu)


        if self.proposal_chunks is not None:
            self.proposals = self._scatter(self.proposals, self.proposal_chunks)

    def __getitem__(self, index):
        """
        Returns a tuple containing data
        :param index: Which GPU we're on, or 0 if no GPUs
        :return: If training:
        (image, im_size, img_start_ind, anchor_inds, anchors, gt_boxes, gt_classes, 
        train_anchor_inds)
        test:
        (image, im_size, img_start_ind, anchor_inds, anchors)
        """
        if index not in list(range(self.num_gpus)):
            raise ValueError("Out of bounds with index {} and {} gpus".format(index, self.num_gpus))

        if self.is_rel:
            rels = self.gt_rels
            fwd_rels = self.fwd_rels
            inv_rels = self.inv_rels
            if index > 0 or self.num_gpus != 1:
                rels_i = rels[index] if self.is_rel else None
                fwd_rels_i = fwd_rels[index] if self.is_rel else None
                inv_rels_i = inv_rels[index] if self.is_rel else None
        elif self.is_flickr:
            rels = (self.gt_sents, self.gt_nodes)
            if index > 0 or self.num_gpus != 1:
                rels_i = (self.gt_sents[index], self.gt_nodes[index])
        else:
            rels = None
            rels_i = None
            fwd_rels_i = None
            inv_rels = None

        if self.proposal_chunks is None:
            proposals = None
        else:
            proposals = self.proposals

        if index == 0 and self.num_gpus == 1:    #todo obj_comb_pos is not in use, please delete later
            image_offset = 0
            if self.is_train:
                return (self.imgs, self.im_sizes[0], image_offset,
                        self.gt_boxes, self.gt_classes, self.src_seq, self.tgt_seq, rels, proposals, self.gt_obj_comb, fwd_rels, inv_rels, self.train_anchor_inds, self.obj_comb_pos, self.norm_boxes, self.obj_rel_mat, self.gt_attr)
            return self.imgs, self.im_sizes[0], image_offset, self.gt_boxes, self.gt_classes, self.src_seq, self.tgt_seq, rels, proposals, self.gt_obj_comb, fwd_rels, inv_rels, None, self.obj_comb_pos, self.norm_boxes, self.obj_rel_mat, self.gt_attr

        # Otherwise proposals is None
        assert proposals is None

        image_offset = self.batch_size_per_gpu * index
        # TODO: Return a namedtuple
        if self.is_train:
            return (
            self.imgs[index], self.im_sizes[index], image_offset,
            self.gt_boxes[index], self.gt_classes[index], self.src_seq[index], self.tgt_seq[index], rels_i[index], None, self.gt_obj_comb[index], fwd_rels_i[index], inv_rels_i[index], self.train_anchor_inds[index], self.obj_comb_pos[index], self.norm_boxes[index], self.obj_rel_mat[index], self.gt_attr[index])
        return (self.imgs[index], self.im_sizes[index], image_offset,
                self.gt_boxes[index], self.gt_classes[index], self.src_seq[index], self.tgt_seq[index], rels_i[index], None, self.gt_obj_comb[index], fwd_rels_i[index], inv_rels_i[index], None, self.obj_comb_pos[index], self.norm_boxes[index],self.obj_rel_mat[index], self.gt_attr[index])

