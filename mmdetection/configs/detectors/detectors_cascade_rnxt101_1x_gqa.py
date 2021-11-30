_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/gqa_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        conv_cfg=dict(type='ConvAWS'),
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        # rfp_sharing=False, #!
        aspp_out_channels=64, #
        aspp_dilations=(1, 3, 6, 1), #
        # stage_with_rfp=(False, True, True, True), #!
        # interleaved=True, #!
        # mask_info_flow=True, #!
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            style='pytorch')))

load_from = None

# below is config from
# https://github.com/joe-siyuan-qiao/DetectoRS/blob/master/configs/DetectoRS/DetectoRS_mstrain_400_1200_x101_32x4d_40e.py
# cannot use, produces various errors
# model = dict(
#     type='RecursiveFeaturePyramid',
#     rfp_steps=2,
#     rfp_sharing=False,
#     stage_with_rfp=(False, True, True, True),
#     num_stages=3,
#     pretrained='open-mmlab://resnext101_32x4d',
#     interleaved=True,
#     mask_info_flow=True,
#     backbone=dict(
#         type='ResNeXt',
#         depth=101,
#         groups=32,
#         base_width=4,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         conv_cfg=dict(type='ConvAWS'),
#         sac=dict(type='SAC', use_deform=True),
#         stage_with_sac=(False, True, True, True),
#         norm_cfg=dict(type='BN', requires_grad=True),
#         style='pytorch'))