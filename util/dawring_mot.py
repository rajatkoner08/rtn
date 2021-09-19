import numpy as np
import cv2

colorArray = np.random.rand(256)

from MOT_Constants import DATA_DIR
from MOT_Constants import LOG_DIR
from MOT_Constants import CROP_WIDTH
from MOT_Constants import CROP_HEIGHT
from MOT_Constants import CROP_PAD

from re3_utils.util import drawing
from re3_utils.util.mot_util import *

def draw_boxes_wd_id(image, original, prediction = None, out_dir= None, img_no = None, debug= False):
    imageToDraw = image.copy()

    for j, box in enumerate(original):
        bbox = box[1:]
        id = int(box[0])
        if prediction is not None:
            # find predicted box based on ID
            predBox = prediction[np.where((prediction[:, 0]).astype(int) == id)][:, 1:]
        text_size = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_PLAIN, 1, 2)

        center = int(bbox[0] + 5), int(bbox[1] + 5 + text_size[0][1])
        pt2 = bbox[0] + 10 + text_size[0][0], bbox[1] + 10 + text_size[0][1]

        color = cv2.cvtColor(np.uint8([[[colorArray[id%256] * 255, 128, 200]]]),
                             cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(imageToDraw,
                      (int(bbox[0]), int(bbox[1])),
                      (int(pt2[0]), int(pt2[1])), color, 2)
        cv2.rectangle(imageToDraw,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)
        #for new target doesnt exist in prediction
        if prediction is not None:
            if len(predBox) > 0:
                predBox = predBox[0, :]
                drawing.drawRect(imageToDraw, predBox, 1, color)
                # cv2.rectangle(imageToDraw,
                #               (int(predBox[0]), int(predBox[1])),
                #               (int(predBox[2]), int(predBox[3])), color, 2)
        cv2.putText(imageToDraw, str(id), center, cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        if debug:
            if out_dir is not None and img_no is not None:
                cv2.imwrite(out_dir + '/' + str(img_no).rjust(6, '0') + '.jpg', imageToDraw)

    return imageToDraw

#debug and draw images with their boxes and ROI
def drawImagesinBatch(images, rois, labels, batch_size, no_of_target, output_dir,fmap_width,fmap_height, iteration=None, img_no = None, label_type = 'relative', prediction= None,targets=None):
    if img_no is None:
        img_no = 1
    img = images.copy()
    roi = roiToImage(rois, fmap_height=fmap_height, fmap_width=fmap_width)
    img = img.reshape((batch_size, 2, img.shape[1], img.shape[2], img.shape[3]))
    roi = roi.reshape(batch_size, 2, no_of_target, 4)
    if  label_type == 'relative':
        label = relativeROItoImage(labels, rois, f_width=fmap_width,
                                   f_height=fmap_height, img_width=CROP_WIDTH, img_height=CROP_HEIGHT,
                                   batchSize=batch_size, no_of_targets=no_of_target)
        if prediction is not None:
            preds = relativeROItoImage(prediction, rois, f_width=fmap_width,
                                       f_height=fmap_height, img_width=CROP_WIDTH, img_height=CROP_HEIGHT,
                                       batchSize=batch_size, no_of_targets=no_of_target)
        if targets is not None:
            trgts = relativeROItoImage(targets, rois, f_width=fmap_width,
                                       f_height=fmap_height, img_width=CROP_WIDTH, img_height=CROP_HEIGHT,
                                       batchSize=batch_size, no_of_targets=no_of_target, type='targets')
    else:
        label = normalizedLabelToOriginal(labels, CROP_WIDTH, CROP_HEIGHT)
        if prediction is not None:
            preds = normalizedLabelToOriginal(prediction, CROP_WIDTH, CROP_HEIGHT)

    label = label.reshape(batch_size, no_of_target, 4)
    if prediction is not None:
        preds = preds.reshape(batch_size, no_of_target, 4)
    if targets is not None:
        trgts = trgts.reshape(batch_size, no_of_target, 4)

    for i in range(0, batch_size):
        d_img0 = img[i, 0, ...].astype(np.uint8)
        d_img1 = img[i, 1, ...].astype(np.uint8)
        for j, box in enumerate(label[i, ...]):
            noisy_roi_box = roi[i, 1, j, ...]
            roi_box = roi[i, 0, j, ...]

            drawing.drawRect(d_img1, box, 1, [0, 0, 255])
            drawing.drawRect(d_img0, roi_box, 1, [0, 128, 128])
            #drawing.drawRect(d_img1, roi_box, 1, [128, 0, 128])
            drawing.drawRect(d_img1, noisy_roi_box, 1, [0, 128, 128])
            if prediction is not None:
                pred = preds[i, j, ...]
                drawing.drawRect(d_img1, pred, 1, [0, 255, 0])
            if targets is not None:
                trgt = trgts[i, j, ...]
                drawing.drawRect(d_img0, trgt, 1, [0, 0, 255])
            plots = [d_img0, d_img1]
            subplots = drawing.subplot(plots, 1, 2, outputWidth=CROP_WIDTH, outputHeight=CROP_HEIGHT)
        if iteration is not None:
            img_name = str(iteration)+'_'+str(img_no)
        else:
            img_name = str(img_no)
        cv2.imwrite(output_dir +'/'+ img_name.rjust(6, '0') + '.jpg', subplots)
        img_no += 1
    if img_no is not None:
        return img_no

def drawBoxes(image,label_boxes, pred_boxes, img_width, img_height, img_no, out_path, fmap_width, fmap_height):
    # generate label values to calculate loss
    roiBoxes = getROIs(label_boxes, img_width, img_height, CROP_PAD,
                       feature_map_height=fmap_height, feature_map_width=fmap_width)
    rois = np.zeros((2, len(label_boxes), 5))
    for j in range(len(label_boxes)):
        rois[0, j, ...] = np.asarray(([*[0], *roiBoxes[j]]))
        rois[1, j, ...] = np.asarray(([*[1], *roiBoxes[j]]))
    rois = rois.reshape((rois.shape[0] * rois.shape[1], rois.shape[2]))
    labels = imageToRelativeROI(rescaleBBox(label_boxes, img_width, img_height), rois, fmap_width=fmap_width,
                                fmap_height=fmap_height,
                                batchSize=1, no_of_targets=len(label_boxes))

    img = image.copy()
    outputBox = relativeROItoImage(pred_boxes, rois, f_width=fmap_width,
                                   f_height=fmap_height, img_width=CROP_WIDTH, img_height=CROP_HEIGHT,
                                   batchSize=1, no_of_targets=len(pred_boxes))
    roi = roiToImage(rois, fmap_height=fmap_height, fmap_width=fmap_width, img_width=img_width, img_height=img_height)
    roi = roi.reshape(2, len(pred_boxes), 4)

    outputBox = rescaleBBox(outputBox, CROP_WIDTH, CROP_HEIGHT, toScaleWidth=img_width,
                            toScaleHeight=img_height)



    d_img1 = image.astype(np.uint8).copy()
    for j, box in enumerate(outputBox):
        roi_box = roi[0, j, ...]
        l_box = label_boxes[j,...]
        drawing.drawRect(d_img1, box, 1, [255, 255, 0])
        drawing.drawRect(d_img1, l_box, 1, [0, 255, 255])
        drawing.drawRect(d_img1, roi_box, 1, [0, 128, 128])
        plots = [d_img1]
        subplots = drawing.subplot(plots, 1, 1, outputWidth=img_width, outputHeight=img_height)
    img_name = str(img_no)
    cv2.imwrite(out_path + '/' + img_name.rjust(6, '0') + '.jpg', subplots)

    return labels,outputBox