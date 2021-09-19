import numpy as np
import numbers
from PIL import ImageFont,Image

from config import  IM_SCALE
# from MOT_Constants import  CROP_HEIGHT
# from MOT_Constants import  IMG_WIDTH
# from MOT_Constants import  IMG_HEIGHT
# from MOT_Constants import min_iou_self_pred
# from MOT_Constants import CROP_PAD
# from MOT_Constants import HOMO_ROI
# from util import IOU
from util import bb_util

AREA_CUTOFF = 0.4
LIMIT = 99999999

def rescaleBBox(target_bboxes, w, h, scale = IM_SCALE):
    """

    :param target_bboxes: BBox is a ndarray of [x1, y1, x2, y2]
    :param w: original img width
    :param h: original img height
    :param scale: to scale of img
    :return:
    """

    img_scale_factor = scale / max(w, h)
    if h > w:
        im_size = (scale , int(w * img_scale_factor), img_scale_factor)
    elif h < w:
        im_size = (int(h * img_scale_factor), scale, img_scale_factor)
    else:
        im_size = (scale, scale, img_scale_factor)

    width_scale_factor = im_size[1] / w
    height_scale_factor = im_size[0] / h
    bboxes = target_bboxes.copy()

    bboxes[:, 0] *= width_scale_factor
    bboxes[:, 1] *= height_scale_factor
    bboxes[:, 2] *= width_scale_factor
    bboxes[:, 3] *= height_scale_factor

    return bboxes, im_size

# @bboxes {np.array} 4xn array of boxes to be scaled
# @scalars{number or arraylike} scalars for width and height of boxes
# @in_place{bool} If false, creates new bboxes.
def scale_bbox(bbox, scalars,
               clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
               round=False, in_place=False):
    addedAxis = False
    if isinstance(bbox, list):
        bbox = np.array(bbox, dtype=np.float32)
    if isinstance(scalars, numbers.Number):
        scalars = np.full((2, bbox.shape[1]), scalars, dtype=np.float32)
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars, dtype=np.float32)

    bbox = bbox.astype(np.float32)

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    xMid = (bbox[0] + bbox[2]) / 2.0
    yMid = (bbox[1] + bbox[3]) / 2.0
    if not in_place:
        bboxesOut = bbox.copy()
    else:
        bboxesOut = bbox

    bboxesOut[0] = xMid - width * scalars[2] / 2.0
    bboxesOut[1] = yMid - height * scalars[2] / 2.0
    bboxesOut[2] = xMid + width * scalars[2] / 2.0
    bboxesOut[3] = yMid + height * scalars[2] / 2.0

    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if round:
        bboxesOut = np.round(bboxesOut).astype(np.int32)
    return bboxesOut


# BBoxes are ndarray of [x1, y1, x2, y2]
def mirroredBoxes(target_bboxes, img_width):
    bboxes = target_bboxes.copy()
    bboxes[:, 0] = img_width - bboxes[:, 0]
    bboxes[:, 2] = img_width - bboxes[:, 2]

    return bboxes[:,[2, 1, 0, 3]]

# BBoxes are [x1, y1, x2, y2]
def clip_bbox(bbox, minClip, maxXClip, maxYClip):
    bboxesOut = bbox
    addedAxis = False
    if len(bboxesOut.shape) == 1:
        addedAxis = True
        bboxesOut = bboxesOut[:,np.newaxis]
    bboxesOut[[0,2],...] = np.clip(bboxesOut[[0,2],...], minClip, maxXClip)
    bboxesOut[[1,3],...] = np.clip(bboxesOut[[1,3],...], minClip, maxYClip)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    return bboxesOut


def draw_box(draw, boxx, cls_ind, text_str, width=2):
    """

    :param draw: image object
    :param boxx: will be in (xmin,ymin, xmax, ymax)
    :param cls_ind:
    :param text_str: class id or category
    :return:
    """
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=width)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=width)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=width)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=width)

    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    # print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
    #     h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw

def text_wrap(draw, text_list, max_width=IM_SCALE, color=(255, 128, 0, 255)):
    # If the width of the text is smaller than image width
    # we don't need to split it, just add it to the lines array
    # create ',' seperated line
    full_text = ''
    lines = []
    for triplet in text_list:
        full_text = full_text + triplet[0] + '-' + triplet[1] + '-' + triplet[2] + ','

    # create the ImageFont instance
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 18)


    if font.getsize(full_text)[0] <= max_width:
        lines.append(full_text)
    else:
        # split the line by spaces to get words
        words = full_text.split(',')
        i = 0
        # append every word to a line while its width is shorter than image width
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            # when the line gets longer than the max width do not append the word,
            # add the line to the lines array
            lines.append(line+',')

    line_height = font.getsize('hg')[1]
    #start writing from buttom of the img
    x = 0
    y = IM_SCALE-len(lines)*line_height
    for line in lines:
        # draw the line on the image
        draw.text((x, y), line, fill=color, font=font)
        # update the y position so that we can use it for next line
        y = y + line_height

    return draw


# #BBoxes are ndarray of [batchno, x1, y1, x2, y2]
# def getROIs(target_boxes, img_width, img_height, padScale, feature_map_width, feature_map_height):
#
#     #rescale boxes as per CROP size
#     boxes = rescaleBBox(target_boxes, img_width, img_height)
#
#     width = boxes[:, 2] - boxes[:, 0]
#     height = boxes[:, 3] - boxes[:, 1]
#
#     #for homogenious roi resize to roi_pool aspect ratio
#     if HOMO_ROI:
#         asr = np.round(width/height,1)
#         for i in range(len(width)):
#             if asr[i] > 0.5 and asr[i]<2.5:
#                 if asr[i] == 1.5:
#                     break
#                 elif asr[i] >1 : #width is big
#                     factor = (width[i] / 3 - width[i] % 3)
#                     if asr[i] >= 1.85: #if width is too big reduces a bit
#                         factor = np.round(factor * 0.85)
#                     width[i] = 3*factor
#                     height[i] = 2*factor
#
#                 else: #more height
#                     factor = np.round((height[i]/2 - height[i]%2)*.85)
#                     width[i] = 3*factor
#                     height[i] = 2*factor
#
#     xC = (boxes[:, 0] + boxes[:, 2]) / 2
#     yC = (boxes[:, 1] + boxes[:, 3]) / 2
#
#     #maximize boxes based on pad scale
#     rois = np.zeros(shape=(len(boxes), 4))
#     rois[:, 0] = (xC - padScale * width / 2)
#     rois[:, 1] = (yC - padScale * height / 2)
#     rois[:, 2] = (xC + padScale * width / 2)
#     rois[:, 3] = (yC + padScale * height / 2)
#
#     #rescale ROIS to feature map scale
#     rois = rescaleBBox(rois.astype(np.float64),CROP_WIDTH, CROP_HEIGHT,feature_map_width,feature_map_height)
#
#     #maximize the feature map in downsample
#     rois[:, 0] = np.floor(np.fmax(rois[:, 0], 0))
#     rois[:, 1] = np.floor(np.fmax(rois[:, 1], 0))
#     rois[:, 2] = np.ceil(np.fmin(rois[:, 2], feature_map_width))
#     rois[:, 3] = np.ceil(np.fmin(rois[:, 3], feature_map_height))
#
#     return rois
#
# def roiToImage(rois,fmap_width, fmap_height,img_width=CROP_WIDTH,img_height=CROP_HEIGHT):
#     if rois.shape[1] == 5:
#         boxes = rescaleBBox(rois[:,1:].astype(np.float64), img_width=fmap_width, img_height=fmap_height, toScaleHeight=img_height, toScaleWidth=img_width)
#     else:
#         boxes = rescaleBBox(rois.astype(np.float64), img_width=fmap_width, img_height=fmap_height, toScaleHeight=img_height, toScaleWidth=img_width)
#     return boxes
#
# #convert img coordinate to relative ROI based coordinate
# def imageToRelativeROI(targetBoxes, ROIs, fmap_width, fmap_height, batchSize = None, no_of_targets = None, type = 'label'):
#
#     bboxes = targetBoxes.copy()
#     if batchSize is not None and no_of_targets is not None:
#         if type == 'label':
#             ROIs = ROIs.reshape(batchSize, 2, no_of_targets, 5)
#             ROIs = ROIs[:, 1, :].reshape(batchSize * no_of_targets, 5)
#         else:
#             ROIs = ROIs.reshape(batchSize, 2, no_of_targets, 5)
#             ROIs = ROIs[:, 0, :].reshape(batchSize * no_of_targets, 5)
#
#     #convert ROI to image coordinate
#     roiBoxesinImg = roiToImage(ROIs, fmap_width=fmap_width, fmap_height=fmap_height)
#
#     roiWidth = roiBoxesinImg[:, 2] - roiBoxesinImg[:, 0]
#     roiHeight = roiBoxesinImg[:, 3] - roiBoxesinImg[:, 1]
#
#     #convert to relative ROI coordinate
#     bboxes[:, 0] -= roiBoxesinImg[:, 0]
#     bboxes[:, 1] -= roiBoxesinImg[:, 1]
#     bboxes[:, 2] -= roiBoxesinImg[:, 0]
#     bboxes[:, 3] -= roiBoxesinImg[:, 1]
#
#     #to avoid devide by 0 error
#     for i in range(0, len(roiWidth)):
#         if roiHeight[i] > 0 and roiWidth[i] > 0:
#             bboxes[i, 0] /= roiWidth[i]
#             bboxes[i, 1] /= roiHeight[i]
#             bboxes[i, 2] /= roiWidth[i]
#             bboxes[i, 3] /= roiHeight[i]
#
#     return bboxes*10
#
#
# def relativeROItoImage(targetBoxes, ROIs, f_width, f_height, img_width, img_height, batchSize=None, no_of_targets=None, type='label'):
#
#     bboxes = targetBoxes.copy()
#     bboxes /= 10
#
#     if batchSize is not None and no_of_targets is not None:
#         if type == 'label':
#             ROIs = ROIs.reshape(batchSize, 2, no_of_targets, 5)
#             ROIs = ROIs[:, 1, :].reshape(batchSize * no_of_targets, 5)
#         else:
#             ROIs = ROIs.reshape(batchSize, 2, no_of_targets, 5)
#             ROIs = ROIs[:, 0, :].reshape(batchSize * no_of_targets, 5)
#
#     roiBoxesinImg = roiToImage(ROIs, fmap_width=f_width, fmap_height=f_height,img_width=img_width, img_height= img_height)
#
#     roiWidth = roiBoxesinImg[:, 2] - roiBoxesinImg[:, 0]
#     roiHeight = roiBoxesinImg[:, 3] - roiBoxesinImg[:, 1]
#
#     bboxes[:, 0] *= roiWidth
#     bboxes[:, 1] *= roiHeight
#     bboxes[:, 2] *= roiWidth
#     bboxes[:, 3] *= roiHeight
#
#     # convert to relative ROI coordinate
#     bboxes[:, 0] += roiBoxesinImg[:, 0]
#     bboxes[:, 1] += roiBoxesinImg[:, 1]
#     bboxes[:, 2] += roiBoxesinImg[:, 0]
#     bboxes[:, 3] += roiBoxesinImg[:, 1]
#
#     return bboxes
#
#
# def iou_distance_matrix(objs, hyps, max_iou=1.):
#     """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.
#     """
#     #conver x1,y1,x2,y2 format to z1,y1,w,h
#     objs[:,2]= objs[:,2]-objs[:,0]
#     objs[:,3] = objs[:,3] - objs[:,1]
#
#     hyps[:, 2] = hyps[:, 2] - hyps[:, 0]
#     hyps[:, 3] = hyps[:, 3] - hyps[:, 1]
#
#
#     objs = np.atleast_2d(objs).astype(float)
#     hyps = np.atleast_2d(hyps).astype(float)
#
#     if objs.size == 0 or hyps.size == 0:
#         return np.empty((0, 0))
#
#     assert objs.shape[1] == 4
#     assert hyps.shape[1] == 4
#
#     br_objs = objs[:, :2] + objs[:, 2:]
#     br_hyps = hyps[:, :2] + hyps[:, 2:]
#
#     C = np.empty((objs.shape[0], hyps.shape[0]))
#
#     for o in range(objs.shape[0]):
#         for h in range(hyps.shape[0]):
#             isect_xy = np.maximum(objs[o, :2], hyps[h, :2])
#             isect_wh = np.maximum(np.minimum(br_objs[o], br_hyps[h]) - isect_xy, 0)
#             isect_a = isect_wh[0] * isect_wh[1]
#             union_a = objs[o, 2] * objs[o, 3] + hyps[h, 2] * hyps[h, 3] - isect_a
#             if union_a != 0:
#                 C[o, h] = 1. - isect_a / union_a
#             else:
#                 C[o, h] = np.nan
#
#     C[C > max_iou] = np.nan
#     return C
#
# #normalized label
# def normalizedLabel(label, img_width, img_height):
#     label_copy =  label.copy()
#     label_copy[:,0] /= img_width
#     label_copy[:,1] /= img_height
#     label_copy[:,2] /= img_width
#     label_copy[:,3] /= img_height
#
#     return label_copy*10
#
# #normalized label
# def normalizedLabelToOriginal(label, img_width, img_height):
#     label_copy = label.copy()
#     label_copy /= 10
#
#     label_copy[:, 0] *= img_width
#     label_copy[:, 1] *= img_height
#     label_copy[:, 2] *= img_width
#     label_copy[:, 3] *= img_height
#
#     return label_copy
#
# # Randomly jitter the box for a bit of noise.
# def add_noise(bbox, prevBBox, imageWidth, imageHeight):
#     numTries = 0
#     bboxXYWHInit = bb_util.xyxy_to_xywh(bbox)
#     while numTries < 10:
#         bboxXYWH = bboxXYWHInit.copy()
#         centerNoise = np.random.laplace(0,1.0/5,2) * bboxXYWH[[2,3]]
#         sizeNoise = np.clip(np.random.laplace(1,1.0/15,2), .6, 1.4)
#         bboxXYWH[[2,3]] *= sizeNoise
#         bboxXYWH[[0,1]] = bboxXYWH[[0,1]] + centerNoise
#         if not (bboxXYWH[0] < prevBBox[0] or bboxXYWH[1] < prevBBox[1] or
#             bboxXYWH[0] > prevBBox[2] or bboxXYWH[1] > prevBBox[3] or
#             bboxXYWH[0] < 0 or bboxXYWH[1] < 0 or
#             bboxXYWH[0] > imageWidth or bboxXYWH[1] > imageHeight):
#             numTries = 10
#         else:
#             numTries += 1
#
#     return fix_bbox_intersection(bb_util.xywh_to_xyxy(bboxXYWH), prevBBox, imageWidth, imageHeight)
#
# # Make sure there is a minimum intersection with the ground truth box and the visible crop.
# def fix_bbox_intersection(bbox, gtBox, imageWidth, imageHeight):
#     if type(bbox) == list:
#         bbox = np.array(bbox)
#     if type(gtBox) == list:
#         gtBox = np.array(gtBox)
#     #bug fixes from Datagenerator as its pass as string
#     if not  gtBox.dtype == 'float64' or gtBox.dtype == 'float32' or gtBox.dtype == 'int':
#        gtBox = np.array(gtBox).astype(np.float32)
#
#     gtBoxArea = float((gtBox[3] - gtBox[1]) * (gtBox[2] - gtBox[0]))
#     bboxLarge = getROIs(np.expand_dims(bbox, axis=0), IMG_WIDTH, IMG_HEIGHT, CROP_PAD,
#                           feature_map_height=IMG_HEIGHT, feature_map_width=IMG_WIDTH)[0,:]
#     itr = 0
#     while IOU.intersection(bboxLarge, gtBox) / gtBoxArea < min_iou_self_pred:
#         bbox = bbox * .9 + gtBox * .1
#         bboxLarge = getROIs(np.expand_dims(bbox, axis=0), IMG_WIDTH, IMG_HEIGHT, CROP_PAD,
#                             feature_map_height=IMG_HEIGHT, feature_map_width=IMG_WIDTH)[0,:]
#         itr += 1
#         if itr > 5:
#             bbox = gtBox
#             break
#     return bbox
#
# #get dynamic roi pool size based i=on image size
# def getFeatureMapSize(imgWidth, imgHeight, network):
#     #all feature map size will be 1st the main, then skips
#     if  imgWidth == 960 and imgHeight==540:
#         if network == 'alexnet':
#             return [58, 32, 118, 66]
#         elif network == 'vgg16':
#             return  [60, 34, 120, 68]
#     if imgWidth == 480 and imgHeight==272:
#         if network == 'alexnet':
#             return [28, 15, 58, 32]
#         elif network == 'vgg16':
#             return  [30, 17, 68, 34]
