"""
Let's get the relationships yo
"""

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
from lib.pytorch_misc import enumerate_by_image,append_and_pad, get_feats_size, get_index_pos
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM, DETECTOR_CONF, NumAttr

from lib.transformer.Models import Encoder, Decoder, Rel
from lib.fpn.box_utils import draw_obj_box
from lib.pytorch_misc import get_obj_comb_by_batch, init_weights
from lib.relation_prediction import Relation_Prediction
from lib.fpn.box_utils import bbox_overlaps, center_size, increase_relative_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import make_one_hot, get_normalized_rois, to_onehot, get_combined_feats, Flattener
#from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.helper import get_fwd_inv_rels
from lib.word_vectors import obj_edge_vectors
from torchvision.ops.roi_align import roi_align as RoIAlignFunction

from lib.draw_attention import process_node_edge_map, process_obj_attn_map, process_edge_attn_map
import copy

MODES = ('sgdet', 'sgcls', 'predcls')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, rel_classes, config, mode='sgdet',
                 hidden_dim=256, nl_obj=2, nl_edge=2, dropout_rate=0.2, use_temporal_conn=True, proj_share_weight = True,
                 embs_share_weight = True, dropout = 0.1, gap_dim=512):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode
        self.batch_size = config.batch_size
        self.use_obj_index = config.obj_index_enc
        self.use_word_emb = config.use_word_emb

        self.nl_obj = config.nl_obj
        self.nl_edge = config.nl_edge
        self.pooling_dim = config.pooling_dim
        self.pos_dim = config.pos_dim
        self.normalized_roi = config.normalized_roi
        self.hidden_dim = config.attn_dim
        self.union_boxes = config.union_boxes
        self.spatial_box = config.spatial_box
        self.d_k = self.d_v = config.attn_dim // config.n_head
        self.pass_obj_feats_to_classifier = config.pass_obj_feats_to_classifier
        self.use_background = config.use_bg
        self.bg_max_pool = torch.nn.MaxPool2d(7, stride=3)       #use large receptive field for bg
        self.max_positional_obj = config.max_token_seq_len
        self.use_sub_obj_index = config.highlight_sub_obj
        self.use_nm_baseline = config.use_nm_baseline
        self.dropout = nn.Dropout(config.dropout)
        self.e2e = config.edge2edge_attn
        self.pass_obj_dist = config.pass_obj_dist
        self.attn4rel = config.use_obj_rel_map
        self.edge_loss = config.use_valid_edges
        self.use_extra_pos = config.use_extra_pos
        self.comb_edge = config.comb_edge
        self.use_gap = config.use_gap
        self.attr = config.use_attr
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid() # initialize sigmoid layer

        #assert (self.use_word_emb == True and self.pass_obj_dist ==False),'Pass obj dist only if word embedding is used' #todo use proper assert
        if self.spatial_box:
            self.box_conv = nn.Sequential(
                nn.Conv2d(1, dim // 2, kernel_size=7, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(dim // 2, momentum=BATCHNORM_MOMENTUM),
                nn.Conv2d(dim // 2, dim//2, kernel_size=5, stride=2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(dim//2, momentum=BATCHNORM_MOMENTUM),
                nn.Conv2d(dim // 2, dim//2, kernel_size=3, stride=2,padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(dim//2, momentum=BATCHNORM_MOMENTUM),
            );

        # GLOVE vector EMBEDDINGS #todo will be replaced by BERT emebedding
        self.embed_dim = config.bert_embd_dim
        if self.use_word_emb:
            embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
            self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
            self.obj_embed.weight.requires_grad = False
            self.obj_embed.weight.data = embed_vecs.clone()
            self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
            self.obj_embed2.weight.requires_grad = False
            self.obj_embed2.weight.data = embed_vecs.clone()

        #self.layer_norm= nn.LayerNorm(config.attn_dim)
        if self.edge_loss:
            assert self.hidden_dim == config.attn_dim #todo check compatibility wd above similar assigment
            self.correct_edge = nn.Sequential(*[nn.Linear(self.hidden_dim, 1),nn.Sigmoid()]) #nn.Linear(self.hidden_dim+config.attn_dim, self.hidden_dim),nn.Dropout(dropout_rate),
            self.correct_edge.apply(init_weights)
            #self.correct_edge._modules['0'].bias.data.fill_(-0.881657035)

        if self.use_gap:
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=37)  # todo replcae wd correct param

        # determine feature dimension
        node_dim, edge_dim = get_feats_size(self.pooling_dim, self.use_word_emb, self.normalized_roi, self.union_boxes, self.spatial_box, use_gap=self.use_gap,
                                            gap_dim=gap_dim, num_classes=self.num_classes)
        self.compress_node = nn.Linear(node_dim, config.attn_dim)
        torch.nn.init.xavier_normal(self.compress_node.weight, gain=1.0)
        if node_dim == edge_dim:
            self.compress_edge = self.compress_node
        else:
            self.compress_edge = nn.Linear(edge_dim, config.attn_dim)
            torch.nn.init.xavier_normal(self.compress_edge.weight, gain=1.0)

        if not self.normalized_roi:
            self.pos_embed = nn.Sequential(*[
                nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),#
                nn.Linear(4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ])
        else:
            self.pos_embed = None

        # whether to pass object detector features along with context
        if self.pass_obj_feats_to_classifier :
            self.obj_out = nn.Linear(self.pooling_dim + config.attn_dim, self.num_classes)
        else:
            self.obj_out = nn.Linear(config.attn_dim, self.num_classes)
        torch.nn.init.xavier_normal(self.obj_out.weight, gain=1.0)

        #for attribute classification
        if self.attr:
            self.attr_linear = nn.Linear(config.attn_dim + self.num_classes + NumAttr, NumAttr) # + NumAttr todo attribute change here

        if self.use_nm_baseline:
            self.decoder_lin = nn.Linear(self.pooling_dim + self.embed_dim + 128, self.num_classes)


        if self.nl_obj > 0:
            self.obj_ctx_enc = Encoder(self.num_classes, max_seq=self.max_positional_obj+1, d_word_vec=config.attn_dim,
                                       d_model=config.attn_dim, d_inner=config.fwd_dim, n_layers=config.nl_obj,
                                       n_head=config.n_head, d_k=self.d_k, d_v=self.d_v, custom_pe= self.use_sub_obj_index, dropout=config.dropout)

        if self.nl_edge > 0:
            self.edge_ctx_dec = Decoder(self.num_rels, max_seq=self.max_positional_obj+1 if self.use_sub_obj_index else 250, d_word_vec=config.attn_dim,
                                        d_model=config.attn_dim, d_inner=config.fwd_dim, n_layers=config.nl_edge,n_head=config.n_head,
                                        d_k=self.d_k, d_v=self.d_v, custom_pe= self.use_sub_obj_index, e2e=self.e2e, dropout=config.dropout)
            # self.edge_ctx_dec = nn.Linear(self.pooling_dim+5, config.attn_dim)
            # self.edge_ctx_dec.weight = torch.nn.init.xavier_normal(self.edge_ctx_dec.weight, gain=1.0)
            # self.edge_ctx_dec.bias.data.zero_()
        if self.attn4rel:
            self.node_rel_ctx = Rel(d_model=config.attn_dim, dropout=config.dropout)
    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)
    def forward(self, bg_fmap, obj_fmaps, obj_logits, obj_comb, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None, src_seq=None, tgt_seq=None, index_enc=None,
                norm_boxes=None,  union_boxes_feats = None, u_rois=None, attir=None, od_atrr_dists=None, return_attn=True):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """
        obj_embed = F.softmax(obj_logits, dim=1)
        if self.use_word_emb :
            obj_embed = obj_embed @ self.obj_embed.weight

        obj_word_embed = obj_embed      #incase nm_baseline is not used
        if self.use_nm_baseline:
            obj_pre_reps = torch.cat((obj_fmaps, obj_embed, self.pos_embed(center_size(box_priors))), 1)
            obj_embed = self.decoder_lin(obj_pre_reps)
            if self.use_word_emb :
                obj_word_embed = self.dropout(obj_embed @ self.obj_embed.weight)

        pos_feats = norm_boxes[:,1:] if self.normalized_roi else box_priors

        if self.spatial_box:
            spt_obj_boxes = draw_obj_box(box_priors.clone(), pooling_size=bg_fmap.shape[-1], box_seq=src_seq[:,1].clone()) -0.5
            obj_fmaps += self.box_conv(spt_obj_boxes).view(obj_fmaps.shape[0],-1)
            if self.union_boxes:
                spt_pred_boxes = draw_obj_box(u_rois.clone(), pooling_size=bg_fmap.shape[-1], box_seq=tgt_seq[:, 1].clone()) - 0.5
                union_boxes_feats += self.box_conv(spt_pred_boxes).view(union_boxes_feats.shape[0],-1)
        if self.use_gap:
            gap_feats = self.avg_pool(bg_fmap).view(bg_fmap.shape[0], -1)[im_inds]
        obj_combined_feats = get_combined_feats(obj_fmaps, obj_word_embed, pos_feats, self.compress_node, self.normalized_roi, self.pos_embed,
                                                self.dropout(obj_embed) if self.pass_obj_dist else  None, spatial_box=self.spatial_box, gap=gap_feats if self.use_gap else None)
        obj_index = None

        src_seq = src_seq[:, 1].view(self.batch_size, src_seq.shape[0] // self.batch_size)
        tgt_seq = tgt_seq[:, 1].view(self.batch_size, tgt_seq.shape[0] // self.batch_size)
        enc_obj_feats = obj_combined_feats.view(self.batch_size, obj_combined_feats.shape[0] // self.batch_size, obj_combined_feats.shape[1])

        if self.use_obj_index:
            obj_index = get_index_pos(src_seq)

        #call multi head attn Encoder, attn map not used now, later on u can
        enc_rep, obj_self_attn = self.obj_ctx_enc(enc_obj_feats, src_seq, obj_index, return_attns=return_attn)  #todo removed slf attn to use attn4rel, later decide

        #call rel layer to use attn for rel prediction
        if self.attn4rel:
            obj_self_attn = self.node_rel_ctx(enc_rep, src_seq)
        # else:
        #     obj_self_attn = None

        if self.use_background:
            flattend_enc_rep = enc_rep[:, :-1, :].contiguous().view(enc_rep.shape[0] * (enc_rep.shape[1] - 1), -1)  # todo is it helpful?
        else:
            flattend_enc_rep = enc_rep.view(enc_rep.shape[0] * enc_rep.shape[1], -1)

        obj_dists = self.obj_out(torch.cat((obj_fmaps, flattend_enc_rep), 1) if self.pass_obj_feats_to_classifier else flattend_enc_rep)
        obj_preds = obj_logits.max(1)[1]

        #for attribute classification todo  for attribute change here
        if self.attr:
            attr_dist = self.sigmoid(self.attr_linear(torch.cat((flattend_enc_rep[attir[:,1]], obj_dists[attir[:, 1]], F.sigmoid(od_atrr_dists[attir[:,1]])), 1)))#
        else:
            attr_dist = None

        # Now use the actual class
        if self.mode=='predcls' :
                assert obj_labels is not None
                obj_preds = obj_labels
                obj_embed2 = self.obj_embed2(obj_labels) if self.use_word_emb else to_onehot(obj_labels,self.num_classes)
        elif self.use_word_emb: #todo change,
                obj_embed2 = self.obj_embed2(obj_preds)
        else:
            obj_embed2 = obj_dists

        pred_feats = get_combined_feats(union_boxes_feats if self.union_boxes else obj_fmaps, self.dropout(obj_embed2), u_rois[:,1:] if self.union_boxes else pos_feats,
                                        self.compress_edge, self.normalized_roi, self.pos_embed, self.dropout(obj_embed) if self.pass_obj_dist else  None,
                                        obj_comb, edge=True, union=self.union_boxes, spatial_box=self.spatial_box)

        if not self.union_boxes and self.use_extra_pos : #todo experiemnt with +/*/-
            pred_feats += union_boxes_feats

        if self.comb_edge:  # for edge coming from two nodes todo do some exp wd ordering
            pred_feats = pred_feats * enc_obj_feats.view(-1,self.hidden_dim)[obj_comb[:, 1]] * enc_obj_feats.view(-1,self.hidden_dim)[obj_comb[:, 2]]

        pred_feats = pred_feats.view(self.batch_size, pred_feats.shape[0] // self.batch_size, pred_feats.shape[1])  # reshape frame wise obj combination #b,#edge,#features
        index_enc = index_enc.view(self.batch_size, index_enc.shape[0] // self.batch_size, index_enc.shape[1])

        if self.use_sub_obj_index:
            index_enc = torch.add(index_enc[:, :, 1:],1) #as positional index of obj is start from 1
        else:
            index_enc = get_index_pos(tgt_seq)

        # now get the context of edge including BG
        if self.nl_edge>0:
            edge_ctx, edge_self_attn, edge_node_attn = self.edge_ctx_dec(pred_feats, tgt_seq, src_seq, enc_rep,
                                                                         index_enc, return_attns=return_attn, e2e=self.e2e)
        else:
            edge_ctx = self.dropout(pred_feats)
            edge_node_attn=None
            edge_self_attn=None

        if self.edge_loss:
            edge_out = self.correct_edge(edge_ctx.view(-1, self.hidden_dim))#torch.cat((,pred_feats.view(-1, self.hidden_dim)),1)
        else:
            edge_out = None

        return obj_dists, obj_preds, flattend_enc_rep, edge_ctx.view(-1, self.hidden_dim), obj_self_attn, edge_self_attn, edge_node_attn, edge_out, attr_dist


class TransformerModel(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, classes, rel_classes, config, filenames=None, save_attn = False, inference=False):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(TransformerModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = config.num_gpus
        self.batch_size = config.batch_size
        assert config.mode in MODES
        self.mode = config.mode
        self.use_background = config.use_bg
        self.combined_feats = True # todo replace wd param
        self.drout = config.dropout
        self.b_dropout = nn.Dropout(2*self.drout)
        self.use_gap = False# todo  fix this laterconfig.use_gap
        self.eval_method =config.eval_method
        self.use_attr = config.use_attr

        self.filenames = filenames
        self.save_attn = save_attn
        self.counter = 0
        self.pooling_size = 7
        self.pool_dim = config.pooling_dim
        self.hidden_dim = config.attn_dim
        self.use_mmdet = config.use_mmdet
        self.fwd_dim = config.fwd_dim
        self.use_union_boxes = config.union_boxes
        self.spo = config.spo
        self.focal_loss = config.use_FL
        self.edge_loss = config.use_valid_edges
        self.use_extra_pos = config.use_extra_pos
        self.refine_obj = config.refine_obj

        self.use_bias = config.use_bias
        self.use_norm_boxes = config.normalized_roi
        self.use_tanh = config.use_tanh
        self.inference = inference
        self.require_overlap = config.require_overlap_det #and self.mode == 'sgdet'  #todo check if sgdet is needed

        @property
        def num_classes(self):
            return len(self.classes)

        self.num_rels = len(self.rel_classes)

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if config.use_proposals else 'refinerels') if config.mode == 'sgdet' else 'gtbox',
            use_resnet=config.use_resnet,
            thresh=DETECTOR_CONF,
            max_per_img=64,
            use_mmdet=config.use_mmdet,
            mmdet_config=config.mmdet_config,
            mmdet_ckpt=config.ckpt,
        )

        self.context = LinearizedContext(self.classes, self.rel_classes, config, mode=self.mode, gap_dim=4096 if config.use_mmdet else 512)

        if config.use_mmdet:
            self.roi_fmap_obj = nn.Sequential(*[nn.Linear(3072,self.pool_dim),nn.Dropout(0.5),nn.ReLU(inplace=True)]) #nn.Conv1d(3*256,256, kernel_size=(1,1)) #for cascade rcnn the number of stage is 3 and out chanel 256
            self.roi_fmap_obj.apply(init_weights)
            if self.use_union_boxes:     #used for union boxes
                #self.reduce_dim = copy.deepcopy(self.reduce_dim_obj)
                self.roi_extractor = copy.deepcopy(self.detector.roi_fmap)
                self.roi_fmap =copy.deepcopy(self.roi_fmap_obj)
        elif config.use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else: #todo there is an extra roi fmap in detector, check how heavy or large it is
            if self.use_union_boxes:     #used for union boxes
                # self.roi_fmap = nn.Sequential(*[nn.Linear(512*7*7, 4096),nn.ReLU(inplace=True), nn.Linear(4096,4096), nn.Dropout(0.5),nn.ReLU(inplace=True)])
                # self.roi_fmap.apply(init_weights)
                roi_fmap = [
                    Flattener(),
                    load_vgg(use_dropout=False, use_relu=False, use_linear=config.pooling_dim == 4096, pretrained=False).classifier,
                ]
                if config.pooling_dim != 4096:
                    roi_fmap.append(nn.Linear(4096, config.pooling_dim))
                self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        ###################################

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        if self.use_union_boxes or self.use_extra_pos:
            self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                                  dim=256 if config.use_mmdet else 512, batch_size=self.batch_size,
                                                  use_uimg_feats=self.use_union_boxes, fwd_dim=self.hidden_dim,
                                                  pool_dim=config.pooling_dim, mmdet=self.use_mmdet,
                                                  roi_extractor=self.roi_extractor, roi_fmap = self.roi_fmap )#reduce_dim = self.reduce_dim if self.use_mmdet else None

        self.nl_edge = config.nl_edge

        #init relation prediction module
        self.rel_prediction = Relation_Prediction(config, self.num_rels)


        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        #self.sub_obj_emb.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / 2*self.hidden_dim))  #todo check like atten weight

        # self.pred_emb = nn.Linear(config.fwd_dim * 2, config.fwd_dim)  #, bias=True
        # nn.init.xavier_normal_(self.pred_emb.weight)
        #self.final_rel.bias.data.fill_(-1.012137945)


        if self.use_bias:
            self.freq_bias = FrequencyBias()


    def visual_rep(self, features, rois, pair_inds, use_norm_boxes, im_sizes, tgt_seq=None):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """

        assert pair_inds.size(1) == 3
        uboxes, u_rois, s_o_rois, o_s_rois = self.union_boxes(features, rois, pair_inds, use_norm_boxes, im_sizes, tgt_seq)
        if self.use_union_boxes and not self.use_mmdet:
            return self.roi_fmap(uboxes), u_rois, s_o_rois, o_s_rois
        else:
            return uboxes, u_rois, s_o_rois, o_s_rois
        #return uboxes.view(uboxes.size(0),-1), u_rois


    def get_rel_inds(self, im_inds, box_priors, gt_rels, mode, im_sizes):
        # Get the possible object combination and relation from batch

        im_inds_batchwise = im_inds.view(self.batch_size, im_inds.shape[0] // self.batch_size).data.cpu().numpy()
        box_priors_batchwise = box_priors.view(self.batch_size, box_priors.shape[0] // self.batch_size,
                                               box_priors.shape[1]).data.cpu().numpy()
        gt_rels_np = gt_rels.data.cpu().numpy()

        src_seq_all = []
        tgt_seq_all = []
        obj_comb_all = []
        fwd_rels_all = []
        inv_rels_all = []
        max_seq = 0
        for i, (im_ind, boxes, im_size) in enumerate(zip(im_inds_batchwise, box_priors_batchwise, im_sizes)):
            valid_seq = np.max(np.where(box_priors_batchwise[i].any(axis=1)))+1 #as index start from 0
            if valid_seq > max_seq:
                max_seq = valid_seq

            if mode in ('predcls', 'sgcls') and self.eval_method=='gt':#:
                gt_rels_image = gt_rels_np[np.where(gt_rels_np[:, 0] == i), :][0, :, :]
                possible_obj_comb = np.unique(np.sort(gt_rels_image[:, 1:3], axis=1), axis=0)
                possible_obj_comb, fwd_rels, inv_rels = get_fwd_inv_rels(possible_obj_comb, gt_rels_image[:,1:], val=True)
            else:
                possible_obj_comb = im_ind[:valid_seq, None] == im_ind[None, :valid_seq]
                np.fill_diagonal(possible_obj_comb, 0)

                # Require overlap for detection
                if self.eval_method=='increase_overlap':
                    boxes = increase_relative_size(boxes, im_size[1], im_size[0], relative_per=0.1)
                boxes_iou = (bbox_overlaps(boxes[:valid_seq, :], boxes[:valid_seq, :]) > 0)
                inv_boxes_iou = np.invert(boxes_iou) #non overlap edge
                np.fill_diagonal(inv_boxes_iou, False)
                non_overlap_edge = np.argwhere(np.triu(inv_boxes_iou))
                np.fill_diagonal(boxes_iou, False)
                overlap_edge = np.argwhere(np.triu(boxes_iou))
                if overlap_edge.shape[0]< 400 and mode in ('predcls', 'sgcls'):
                    numOfSample = min(400-len(overlap_edge),len(non_overlap_edge))
                    non_overlap_edge = random.sample(list(non_overlap_edge),k=numOfSample)
                    if len(non_overlap_edge)>0:
                        possible_obj_comb = np.unique(np.concatenate((overlap_edge,non_overlap_edge),0),axis=0)
                    else :
                        possible_obj_comb =  overlap_edge
                else:
                    possible_obj_comb = overlap_edge
                #     if boxes_iou.any():  #fixed as test contain imgs with no IOU even with multiple object
                #         possible_obj_comb = possible_obj_comb & boxes_iou
                # possible_obj_comb = np.triu(possible_obj_comb)
                # possible_obj_comb = np.argwhere(possible_obj_comb)
                fwd_rels = np.concatenate((np.arange(0, possible_obj_comb.shape[0])[:, None], possible_obj_comb), 1)
                inv_rels = fwd_rels[:, [0, 2, 1]]

            obj_comb_all.append(possible_obj_comb)
            #if fwd_rels is not None:
            fwd_rels_all.append(fwd_rels)
            # else:
            #     print('none',i)
            # if inv_rels is not None:
            inv_rels_all.append(inv_rels)

            # now computes source and tgt sequences
            src_seq_all.append(np.ones(valid_seq))
            tgt_seq_all.append(np.ones(possible_obj_comb.shape[0]))

        return append_and_pad(batch_size=self.batch_size, im_inds=im_inds, obj_comb = obj_comb_all, fwd_rel = fwd_rels_all, inv_rel = inv_rels_all, src_seq= src_seq_all, tgt_seq = tgt_seq_all)

    def forward(self, x, im_sizes, image_offset, gt_boxes=None, gt_classes=None, src_seq=None, tgt_seq=None, gt_rels=None, proposals=None, gt_obj_comb=None,
                fwd_rels = None, inv_rels = None, train_anchor_inds=None, obj_comb_pos = None, norm_boxes=None, obj_rel_mat = None, gt_attr =None, return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals, gt_obj_comb[:,:3].clone(), train_anchor_inds, return_fmap=True)
        assert not result.is_nan()
        result.edge_gt_lbl = gt_obj_comb[:,3]  #gt_obj_comb[torch.nonzero(gt_obj_comb[:,3])] #todo not gonna work wd sgdet
        if isinstance(result.fmap, list):
            result.fmap = [i.detach() for i in result.fmap]
            # temporarily take the first layer
            #result.fmap = [result.fmap[0]]
        else:
            result.fmap = result.fmap.detach()

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors
        result.od_attrs = gt_attr
        gt_attr_index = gt_attr[:,:2].float()

        if self.training and  self.mode == 'sgdet':  #todo chage for sgdet
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        proj_obj_fmap = self.roi_fmap_obj(result.obj_fmap) #self.reduce_dim_obj(result.obj_fmap).view(result.obj_fmap.shape[0],-1))
        if not self.training : #and result.obj_comb is None
           fwd_rels, inv_rels, src_seq, tgt_seq, gt_obj_comb = self.get_rel_inds(im_inds, result.rm_box_priors, gt_rels, self.mode, im_sizes)  #todo need to match if edge loss is there, either here or sg_eval
            #fwd_rels = fwd_rels[:,:4]
            #inv_rels = inv_rels[:,:4]
            #gt_obj_comb = gt_obj_comb[:,:3]   #todo temp code, remove

        # compute fwd and inv edges, also take care of multiple occurrences of a single edge, attribute with enumenrate by image
        result.obj_comb, fwd_rels, inv_rels, result.od_attrs = get_obj_comb_by_batch(gt_obj_comb[:,:3].clone(), im_inds, fwd_rels, inv_rels, result.od_attrs)
        #fwd_rels = fwd_rels[torch.nonzero(fwd_rels[:, 4])]
        #inv_rels = inv_rels[torch.nonzero(inv_rels[:,4])]

        if self.use_union_boxes or self.use_extra_pos:
            vr, u_rois, s_o_rois, o_s_rois = self.visual_rep(result.fmap, rois, result.obj_comb, self.use_norm_boxes, im_sizes, tgt_seq) #todo u_roi will break in norm_boxes used

        # Prevent gradients from flowing back into score_fc from elsewhere
        result.rm_obj_dists, result.obj_preds, obj_ctx, edge_ctx, obj_self_attn, edge_self_attn, edge_node_attn, result.edge_dist, result.attr_dist = self.context(
            result.fmap, proj_obj_fmap, result.rm_obj_dists.detach(), result.obj_comb,
            im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,  #todo here predcls has no meaning(mayb)
            boxes.data, result.boxes_all, src_seq, tgt_seq, gt_obj_comb[:,:3], norm_boxes,
            vr if (self.use_union_boxes or self.use_extra_pos )else None, u_rois if self.use_union_boxes else None, result.od_attrs, result.od_attr_dist)

        if self.filenames is not None and self.save_attn:
            filenames = self.filenames[self.counter * self.batch_size:self.counter * self.batch_size + self.batch_size]

            good_img =  process_node_edge_map(edge_node_attn, np.asarray(self.classes), src_seq, gt_classes, self.batch_size, filenames, tgt_seq, gt_obj_comb[:, :3])
            if len(good_img):
                process_obj_attn_map(obj_self_attn, np.asarray(self.classes), src_seq, gt_classes, self.batch_size, filenames, good_img)
                process_edge_attn_map(edge_self_attn, np.asarray(self.classes), src_seq, gt_classes, self.batch_size, filenames, good_img, obj_comb=gt_obj_comb[:, :3], tgt_seq=tgt_seq)
            self.counter = self.counter+1

        # save predicted and original obj-rel mapping

        result.pred_obj_rel_mat = None #if self.save_attn else obj_self_attn
        result.gt_obj_rel_mat = obj_rel_mat

        if not self.edge_loss:
            result.edge_gt_lbl = None

        self.rel_prediction(result, obj_ctx, edge_ctx, s_o_rois, o_s_rois, fwd_rels, inv_rels)

        #todo testing as of now later to be implemented
        result.obj_rel_map = obj_rel_mat

        if self.use_bias:
            if (self.training and not self.mode == 'sgdet'):# or self.mode == 'predcls': # training use GT label todo proably will also work wd sgdet
            #if self.training and not self.mode =='sgdet':
                result.rel_dists = result.rel_dists + self.b_dropout(self.freq_bias.index_with_labels(torch.stack((
                    result.rm_obj_labels[result.all_rels[:, 2]],
                    result.rm_obj_labels[result.all_rels[:, 3]],), 1)))
            else:
                result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                    result.obj_preds[result.all_rels[:, 2]],
                    result.obj_preds[result.all_rels[:, 3]],), 1))

        if self.focal_loss:
            result.rel_dists = F.softmax(result.rel_dists, dim=1)

        if self.training and not self.inference:
            return result


        if self.use_attr:
            result.attr_dist = torch.cat((gt_attr_index,result.attr_dist),1)
        result.obj_scores =  F.softmax(result.rm_obj_dists, dim=1).max(1)[0]  # removed 1: index
        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = None
            #todo fix this
            #bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors
        if self.focal_loss:
            rel_rep = result.rel_dists #F.softmax(result.rel_dists, dim=1)
        else:
            rel_rep = F.softmax(result.rel_dists, dim=1)

        return filter_dets(self.batch_size, bboxes, result.obj_scores, result.obj_preds, result.all_rels, rel_rep, result.pred_obj_rel_mat,
                            result.gt_obj_rel_mat, result.rm_obj_dists, result.edge_dist, result.edge_gt_lbl, result.attr_dist, inference=self.inference)

    def __getitem__(self, batch, eval_train=False):
        """ Hack to do multi-GPU training"""
        if self.inference:  # to get the exact trainign data
            self.training = batch[1]
            batch = batch[0]
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
