import sys
import os
sys.path.append(os.getcwd())


from dataloaders.video_vrd import VRDDataLoader, V_VRD
import numpy as np
import torch

from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
filter_duplicate_rels=False
train = V_VRD(mode='train')
test = V_VRD(mode='test')
# import cv2
import pickle

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
import torchvision.transforms.functional as F
from tqdm import tqdm
from config import BOX_SCALE, VIDEO_DICT_PATH, VRD_DATA_DIR
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from util.mot_util import rescaleBBox,scale_bbox

# conf = ModelConfig()
# train, val, test = VG.splits(num_val_im=conf.val_size)
# if conf.test:
#     val = test
#
# train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
#                                                batch_size=conf.batch_size,
#                                                num_workers=conf.num_workers,
#                                                num_gpus=conf.num_gpus)
#
def load_unscaled(fn):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    im_scale = 1024.0 / max(image_unpadded.size)

    image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                  resample=Image.BICUBIC)
    return image

font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)


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

# #-------------------------------------------------------------

def show_VRD_Batch(batch_data):
    print('size of batch',batch_data.size)


train_loader, test_loader = VRDDataLoader.splits(train, test, mode='rel',batch_size=2, num_workers=1, num_gpus=1, pad_batch=True)
#train_loader  = DataLoader(train, batch_size=2, shuffle=True, num_workers=1)
with open(os.path.join(VIDEO_DICT_PATH, 'Parsed_Data', 'idx_to_objects.pickle'), 'rb') as handle:
    idx_to_obj_class = pickle.load(handle)
with open(os.path.join(VIDEO_DICT_PATH, 'Parsed_Data', 'idx_to_predicates.pickle'), 'rb') as handle:
    idx_to_pred = pickle.load(handle)

# pathname = os.path.join(VRD_DATA_DIR, 'qualitative')
# if not os.path.exists(pathname):
#     os.mkdir(pathname)
# video = cv2.VideoWriter(os.path.join(pathname,'video.avi'),-1,1,(IM_SCALE,IM_SCALE))


print('Test data length : ',test_loader.__len__())
for index, data in enumerate(test_loader):
    # full prediction result from model
    img = data.imgs.data.cpu().numpy().copy()
    gt_classes = data.gt_classes.data.cpu().numpy().copy()
    gt_tids = data.gt_tids.data.cpu().numpy().copy()
    gt_relations = data.gt_rels.data.cpu().numpy().copy()
    gt_boxes = data.gt_boxes.data.cpu().numpy().copy()
    img_size = data.im_sizes.copy()[0][0]
    #create temp dict for object class and tid mappings
    tid_to_obj_type = {}

    theimg2 = ToPILImage()(data.imgs.data.squeeze())#Image.fromarray(np.transpose(img[0],(1,2,0)).astype(np.uint8))
    draw2 = ImageDraw.Draw(theimg2)
    for i,gt_box in enumerate(gt_boxes):
        obj_type = idx_to_obj_class[gt_classes[i][1]]
        obj_tid = gt_tids[i][1]
        tid_to_obj_type[obj_tid] = obj_type
        draw2 = draw_box(draw2, gt_box,
                         cls_ind=0,
                         text_str=obj_type +'_' + str(obj_tid))

    pathname = os.path.join(VRD_DATA_DIR, 'qualitative')
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    if index%30==0:
        theimg2.save(os.path.join(pathname, str(index)+'.jpg'), quality=100, subsampling=0)
        print('Saved image number {}'.format(index))
    # video.write(np.array(theimg2))
    # if index == 100:
    #     #cv2.destroyAllWindows()
    #     video.release()
    #     sys.exit()
    # with open(os.path.join(pathname, 'shit.txt'), 'w') as f:
    #     f.write('Ground Truth Relation:\n')
    #     for r in gt_relations:
    #         sub = tid_to_obj_type[r[1]]+'_'+str(r[1])
    #         obj = tid_to_obj_type[r[2]]+'_'+str(r[2])
    #         relation = idx_to_pred[r[3]]
    #         f.write('{} - {} - {}\n'.format(sub, relation, obj))



info = next(iter(train_loader))
print('Test')

