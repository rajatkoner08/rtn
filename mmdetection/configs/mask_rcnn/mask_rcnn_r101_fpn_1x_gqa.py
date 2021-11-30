_base_ = './mask_rcnn_r50_fpn_1x_gqa.py'
model = dict(backbone=dict(depth=101))
