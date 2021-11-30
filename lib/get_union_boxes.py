"""
credits to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/network.py#L91
"""

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from lib.fpn.box_utils import normalized_boxes, union_boxes

#from lib.fpn.roi_align_unnormalized.functions.roi_align import RoIAlignFunction
from torchvision.ops.roi_align import roi_align as RoIAlignFunction
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
import numpy as np
from torch.nn.modules.module import Module
from torch import nn
from config import BATCHNORM_MOMENTUM

class UnionBoxesAndFeats(Module):
    def __init__(self, pooling_size=7, stride=16, dim=256, batch_size=4, concat=False, use_uimg_feats=True, fwd_dim = 2048, pool_dim= 4096,
                 mmdet=False, roi_extractor= None, roi_fmap=None):
        """
        :param pooling_size: Pool the union boxes to this dimension
        :param stride: pixel spacing in the entire image
        :param dim: Dimension of the feats
        :param concat: Whether to concat (yes) or add (False) the representations
        """
        super(UnionBoxesAndFeats, self).__init__()
        
        self.pooling_size = pooling_size
        self.stride = stride
        self.batch_size = batch_size
        self.concat = concat
        self.pool_dim = pool_dim
        self.mmdet =  mmdet

        self.dim = dim
        self.use_feats = use_uimg_feats
        stride = [2,2] if self.mmdet else [2,1]


        self.conv = nn.Sequential(
            nn.Conv2d(2, dim//2, kernel_size=7, stride=stride[0], padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim//2, momentum=BATCHNORM_MOMENTUM),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(dim//2, dim, kernel_size=3, stride=stride[1], padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim, momentum=BATCHNORM_MOMENTUM),
        );

        if not self.use_feats: #insted here this can be moved to context model itself
            self.box_draw_feats = nn.Sequential(nn.Linear(dim*7*7, fwd_dim),nn.BatchNorm1d(fwd_dim, momentum=BATCHNORM_MOMENTUM),nn.ReLU(inplace=True),nn.Linear(fwd_dim,fwd_dim), nn.Dropout(0.5))

        self.roi_extractor = roi_extractor
        self.roi_fmap = roi_fmap
    def forward(self, fmap, rois, union_inds, use_norm_boxes, im_sizes, tgt_seq=None):

        u_rois, rel_boxes = union_boxes(rois, union_inds)
        if tgt_seq is not None :  # set zero to all the bogus values
            u_rois[:, 1:] *= (tgt_seq[:, 1][:, None]).float()
            rel_boxes[:, 1:] *= (tgt_seq[:, 1][:, None]).float()


        pair_rois = torch.cat((rois[:, 1:][union_inds[:, 1]], rois[:, 1:][union_inds[:, 2]]), 1).data.cpu().numpy()
        rects_np = draw_union_boxes(pair_rois, u_rois[:,1:].data.cpu().numpy(), tgt_seq.data.cpu().numpy().astype(np.float32), self.pooling_size * 4 - 1) - 0.5
        rects = self.conv(torch.FloatTensor(rects_np).cuda(u_rois.get_device()))
        #now get the union box and box feats
        if  self.mmdet:
            union_pools = self.roi_fmap(self.roi_extractor(fmap, u_rois))
        else:
            union_pools = RoIAlignFunction(fmap, u_rois,output_size=[self.pooling_size, self.pooling_size], spatial_scale=1 / 16,
                                           sampling_ratio=0)
        union_pools  =  union_pools.view(union_pools.shape[0],-1)
        if self.use_feats:
             combined_union_pools = union_pools + rects.view(rects.shape[0],-1)

        if use_norm_boxes:
            widths = torch.cuda.FloatTensor(
                np.repeat(im_sizes[:, 1][:, None], u_rois.shape[0] // self.batch_size, 1)).view(-1)
            heights = torch.cuda.FloatTensor(
                np.repeat(im_sizes[:, 0][:, None], u_rois.shape[0] // self.batch_size, 1)).view(-1)
            u_rois = torch.cat((u_rois[:, 0][:,None], normalized_boxes(u_rois[:, 1:], widths, heights)),1)
        #union_pools, union_rois = union_boxes(fmap, rois, union_inds, pooling_size=self.pooling_size, stride=self.stride)
        if self.concat:
            u_rois = torch.cat((u_rois, rel_boxes),1)

        if self.use_feats:
            return combined_union_pools, u_rois, rel_boxes[:,:4], rel_boxes[:,4:]
        else:
            return  self.box_draw_feats(rects.view(rects.size(0), -1)), u_rois, rel_boxes[:,:4], rel_boxes[:,4:]

        # pair_rois = torch.cat((rois[:, 1:][union_inds[:, 0]], rois[:, 1:][union_inds[:, 1]]),1).data.cpu().numpy()
        ## rects_np = get_rect_features(pair_rois, self.pooling_size*2-1) - 0.5
        # rects_np = draw_union_boxes(pair_rois, self.pooling_size*4-1) - 0.5
        # rects = Variable(torch.FloatTensor(rects_np).cuda(fmap.get_device()), volatile=fmap.volatile)
        # if self.concat:
        #     return torch.cat((union_pools, self.conv(rects)), 1)
        # return union_pools + self.conv(rects), union_rois

# def get_rect_features(roi_pairs, pooling_size):
#     rects_np = draw_union_boxes(roi_pairs, pooling_size)
#     # add union + intersection
#     stuff_to_cat = [
#         rects_np.max(1),
#         rects_np.min(1),
#         np.minimum(1-rects_np[:,0], rects_np[:,1]),
#         np.maximum(1-rects_np[:,0], rects_np[:,1]),
#         np.minimum(rects_np[:,0], 1-rects_np[:,1]),
#         np.maximum(rects_np[:,0], 1-rects_np[:,1]),
#         np.minimum(1-rects_np[:,0], 1-rects_np[:,1]),
#         np.maximum(1-rects_np[:,0], 1-rects_np[:,1]),
#     ]
#     rects_np = np.concatenate([rects_np] + [x[:,None] for x in stuff_to_cat], 1)
#     return rects_np


# def union_boxes(fmap, rois, union_inds, pooling_size=14, stride=16):
#     """
#     :param fmap: (batch_size, d, IM_SIZE/stride, IM_SIZE/stride)
#     :param rois: (num_rois, 5) with [im_ind, x1, y1, x2, y2]
#     :param union_inds: (num_urois, 2) with [roi_ind1, roi_ind2]
#     :param pooling_size: we'll resize to this
#     :param stride:
#     :return:
#     """
#     assert union_inds.size(1) == 2
#     im_inds = rois[:,0][union_inds[:,0]]
#     assert (im_inds.data == rois.data[:,0][union_inds[:,1]]).sum() == union_inds.size(0)
#     union_rois = torch.cat((
#         im_inds[:,None],
#         torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
#         torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]]),
#     ),1)
#
#     # (num_rois, d, pooling_size, pooling_size)
#     union_pools = RoIAlignFunction(pooling_size, pooling_size,
#                                    spatial_scale=1/stride)(fmap, union_rois)
#     return union_pools, union_rois[:,1:]
#
