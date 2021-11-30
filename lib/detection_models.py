"""Script to load mmdetection models [faster-rcnn, etc.]"""

import argparse
import copy
import os
import os.path as osp
import time
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmcv.runner.checkpoint import load_checkpoint
from config import ROOT_PATH


class MMDetFeatures(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x # pyramid feature maps

class MMDetClassifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.roi_head = model.roi_head
        self.stage = -1

    def forward(self, fmap, rois, return_cls=False):
        return self.roi_head.batch_simple_test(fmap, rois, return_cls)

def load_mmdetection(config_file, ckpt_path, gpus=1, gpu_ids=0):
    """
    load a mmdetection model as an object detector
    :param config_file: config file path
    :param ckpt_file: ckpt of the trained model
    :return: a mmdetection model - [feature_maps, classifier]
    """
    # ====== prepare config ====== #
    config = os.path.join(ROOT_PATH, config_file)

    cfg = Config.fromfile(config)

    cfg.work_dir = osp.join('./tmp',
                            osp.splitext(osp.basename(config))[0])
    cfg.ckpt_path = ckpt_path
    if gpu_ids is not None:
        cfg.gpu_ids = gpu_ids
    else:
        cfg.gpu_ids = range(1) if gpus is None else range(gpus)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg._cfg_dict['log_level' ])

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # ====== build model ====== #
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = model.cuda()
    # ===== transfer weight ==== #
    # TODO double check
    logger.info('load checkpoint from %s', ckpt_path)
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        _ = load_checkpoint(model, ckpt_path,
            map_location=lambda storage, loc: storage.cuda(device_id), strict=False, logger=logger)
    else:
        _ = load_checkpoint(model, ckpt_path, 'cpu', strict=False, logger=logger)

    return MMDetFeatures(model), MMDetClassifier(model)

if __name__ == "__main__":
    load_mmdetection()
