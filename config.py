"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import argparse
from enum import Enum
import numpy as np

ROOT_PATH = "/nfs/data3/koner"
DATA_PATH = 'data'#os.path.join(ROOT_PATH, 'data')
VRD_DATA_DIR = os.path.join(DATA_PATH,'vidVRD')
VIDEO_DICT_PATH = os.path.join(VRD_DATA_DIR,'vidVRD-dataset')
FRAME_PATH = os.path.join(VRD_DATA_DIR,'vidVRD-frames')
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VIDEO_DATA_DIR = os.path.join(VRD_DATA_DIR,'vidVRD-videos')
VG_IMAGES = os.path.join(DATA_PATH,'VG_100K')
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
LABEL_DATA_DIR = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

COCO_PATH = '/home/rowan/datasets/mscoco'
# =============================================================================
# =============================================================================
#index of parsed file
FILE_NO = 0
FRAME_NO = 1
IMG_PATH = 2
OBJ_TID_ROIS = 3
RELATIONS = 4
OBJ_CLASS = 5
#==============================================================================
MODES = ('sgdet', 'sgcls', 'predcls')

POS_ENCODING_TYPE = ('unnormalized', 'no_enc', 'normalized')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

TEST = 25/200 #test/validation splits

DETECTOR_CONF = 0.3 #detector threshold confidence
REL_CONF = 0.1 #relation threshold confidence

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

#proposed relation threshold
Rel_Threshold = 0.5
EDGE_THRESHOLD = 0.4

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256
BG_EDGE_PER_IMG = 75
FG_EDGE_PER_IMG = 25
BG_EDGE = 2

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

SAVE_TOP =False

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self, file=None):
        """
        Defaults
        """
        self.coco = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.clip = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.optimizer = None
        self.multi_pred=False
        self.cache = None
        self.model = None
        self.use_proposals=False
        self.use_resnet=False
        self.use_tanh=False
        self.pad_batch=None
        self.use_bias = False
        self.num_epochs=None
        self.spo=None
        self.use_warmup=None
        self.use_bg=None
        self.det_ckpt=None
        self.nl_edge=None
        self.nl_obj=None
        self.attn_dim=None
        self.fwd_dim = None
        self.bert_embd_dim=None
        self.n_head=None
        self.max_token_seq_len=None
        self.spatial_box = False
        self.pass_obj_feats_to_classifier = None
        self.run_desc=None
        self.embs_share_weight=None
        self.union_boxes=None
        self.pooling_dim = None
        self.pos_dim=None
        self.dropout = None
        self.require_overlap_det=True
        self.highlight_sub_obj = True
        self.obj_index_enc = True
        self.use_bg_rels = True
        self.normalized_roi = True
        self.num_run = None
        self.g1 = None
        self.g2 = None
        self.g3 = None
        self.g4 = None
        self.count_e_dist = False
        self.use_gap = False
        self.use_word_emb = True
        self.use_FL = True
        self.vg_mini = False
        self.seperate_edge =False
        self.use_extra_pos = False
        self.use_nm_baseline = False
        self.train_obj_roi = False
        self.train_detector = False
        self.reduce_lr_obj_enc = False
        self.reduce_lr =  False
        self.reduce_bg_loss =False
        self.freeze_obj_enc =False
        self.edge2edge_attn = False
        self.pass_obj_dist = False
        self.use_obj_rel_map = False
        self.o_valid = False
        self.comb_edge = False
        self.use_valid_edges = False
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        #now open and wrtie parameter into file
        to_write = open(file, 'w')

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))
            to_write.write(str(x) + ' >>> ' + str(y) + '\n')
        to_write.close()

        self.__dict__.update(self.args)

        # if len(self.ckpt) != 0:
        #     self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        # else:
        #     self.ckpt = None

        # if len(self.cache) != 0:
        #     self.cache = os.path.join(ROOT_PATH, self.cache)
        # else:
        #     self.cache = None

        # if len(self.save_dir) == 0:
        #     self.save_dir = None
        # else:
        #     self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))

        if self.model not in ('motifnet', 'stanford'):
            raise ValueError("Invalid model {}".format(self.model))


        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')
        parser.add_argument('-run_desc', dest='run_desc', help='Describe this run,diff from prev', type=str, default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=0)

        parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

        parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
        parser.add_argument('-r', dest='num_run', help='run number to track', type=int, default=0)
        parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

        parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
        parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
        parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                            default=100)
        parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str, default='sgdet')
        parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                            default='motifnet')
        parser.add_argument('-spo', dest='spo', help='in final features concat all s,p,o/s,o,p',  type=str, default='')
        parser.add_argument('-use_bg', dest='use_bg', help='Use full image as background feature in transformer', action='store_true')
        parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                            default='')
        parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
        parser.add_argument('-opt', dest='optimizer', help='use adam/adabound/SGD(default). Not recommended', type=str)
        parser.add_argument('-test', dest='test', help='test set', action='store_true')
        parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
        parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
        parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
        parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
        parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=2)
        parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
        parser.add_argument('-pos_dim', dest='pos_dim', help='Positional embeding dimension', type=int, default=128)
        parser.add_argument('-fwd_dim', dest='fwd_dim', help='Positional Feed Fwd layer dim', type=int, default=2048)
        parser.add_argument('-attn_dim', dest='attn_dim', help='self attention dim', type=int, default=512)
        parser.add_argument('-bert_embd_dim', dest='bert_embd_dim', help='bert word vec embedding dim', type=int, default=200)#todo temp for matching attn dim == word_vec
        parser.add_argument('-n_head', dest='n_head', help='num of attention head', type=int, default=8)
        parser.add_argument('-max_object', dest='max_token_seq_len', help='num of max object processed for positional encoding', type=int, default=64)
        parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
        parser.add_argument('-spatial_box', dest='spatial_box', action='store_true',default=False)
        parser.add_argument('-pass_obj_feats_to_classifier', dest='pass_obj_feats_to_classifier', action='store_true')
        parser.add_argument('-dropout', dest='dropout', help='transformer dropout to add', type=float, default=0.1)
        parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
        parser.add_argument('-use_union_boxes', dest='union_boxes', action='store_true')
        parser.add_argument('-require_overlap', dest='require_overlap_det', action='store_true')
        parser.add_argument('-normalized_roi', dest='normalized_roi', action='store_true')
        parser.add_argument('-highlight_sub_obj', dest='highlight_sub_obj', action='store_true', default=True)
        parser.add_argument('-obj_index_enc', dest='obj_index_enc', action='store_true', default=True)
        parser.add_argument('-seperate_edge', dest='seperate_edge', action='store_false')
        parser.add_argument('-embs_share_weight', dest='embs_share_weight', action='store_false')
        parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
        parser.add_argument('-use_bg_rels', dest='use_bg_rels', action='store_true')
        parser.add_argument('-use_word_emb', dest='use_word_emb', action='store_true')
        parser.add_argument('-use_FL', dest='use_FL', action='store_true')
        parser.add_argument('-use_warmup', dest='use_warmup', action='store_true')
        parser.add_argument('-train_obj_roi', dest='train_obj_roi', action='store_true')
        parser.add_argument('-train_detector', dest='train_detector', action='store_true')
        parser.add_argument('-reduce_lr_obj_enc', dest='reduce_lr_obj_enc', action='store_true')
        parser.add_argument('-reduce_bg_loss', dest='reduce_bg_loss', action='store_true')
        parser.add_argument('-reduce_lr', dest='reduce_lr',help='reduces the le of od and classifier', action='store_true')
        parser.add_argument('-freeze_obj_enc', dest='freeze_obj_enc', action='store_true')
        parser.add_argument('-use_nm_baseline', dest='use_nm_baseline', action='store_true', default=False)  #todo change dedfault here
        parser.add_argument('-use_edge2edge', dest='edge2edge_attn', action='store_true',  default=False)  # todo change dedfault here
        parser.add_argument('-g1', dest='g1', help='gama value of class loss', type=float, default=1.0)
        parser.add_argument('-g2', dest='g2', help='gama value of rel loss', type=float, default=1.0)
        parser.add_argument('-g3', dest='g3', help='gama value of obj rel mapping loss', type=float, default=1.0)
        parser.add_argument('-g4', dest='g4', help='gama value for correct edge loss', type=float, default=1.0)
        parser.add_argument('-count_edge_dist', dest='count_e_dist', help='give count of pos/neg of edge', action='store_true', default=False)
        parser.add_argument('-use_gap', dest='use_gap', help='use global average pool', action='store_true', default=False)
        parser.add_argument('-pass_obj_dist', dest='pass_obj_dist', help='pass class dist to transformer', action='store_true', default=False)  # todo change dedfault here
        parser.add_argument('-use_obj_rel_map', dest='use_obj_rel_map', help='use attn to map obj into rel', action='store_true', default=False)  # todo change dedfault here
        parser.add_argument('-use_extra_pos', dest='use_extra_pos', help='use extra positional featues', action='store_true', default=False)
        parser.add_argument('-comb_edge', dest='comb_edge', help='combine the features from s,e,o',      action='store_true', default=False)
        parser.add_argument('-vg_mini', dest='vg_mini', help='use vg_mini data of 8K POC', action='store_true', default=False)
        parser.add_argument('-use_valid_edges', dest='use_valid_edges', help='use relation based on valied edges',
                            action='store_true', default=False)
        parser.add_argument('-original_valid', dest='o_valid', help='use original training and validation splits',
                            action='store_true', default=True)
        parser.add_argument('-dataset', dest='dataset', help='dataset \in {vg, vrd}', type=str, default='vg')
        return parser
