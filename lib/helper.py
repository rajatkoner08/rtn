import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lib.fpn.box_utils import center_size,point_form
font = ImageFont.load_default() #ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 16)
import torch
import random
import os

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

def get_unique_rels(rels):
    # return all unique object combinations and their number of occurrences as dict
    all_unique_rel = {}
    for rel in rels:
        if tuple(np.sort(rel)) not in all_unique_rel.keys():
            all_unique_rel[tuple(np.sort(rel))] = 1
        else:
            all_unique_rel[tuple(np.sort(rel))] +=1

    return  all_unique_rel

def intersect_2d_rels(x1, x2):

    aset = set([tuple(x) for x in x1])
    bset = set([tuple(x) for x in x2])

    return np.array([x for x in aset & bset])

def missed_rels(from_rels, to_rels):

    aset = set([tuple(x) for x in from_rels])
    bset = set([tuple(x) for x in to_rels])

    return aset.difference(bset)

def get_decoded_rels(rels, classes, combination, class_name, rels_name):
    all_decoded_rels = []

    for each_comb in combination:
        all_rels = []
        fwd_rels = np.where((rels[:, :2] == each_comb).all(axis=1))[0]
        inv_rels = np.where((rels[:, :2] == each_comb[[1,0]]).all(axis=1))[0]
        if len(fwd_rels)>0:
            all_rels.append(rels[fwd_rels])
        if len(inv_rels)>0:
            all_rels.append(rels[inv_rels])
        for all_rels_sub in all_rels:
            for rel in all_rels_sub:
                all_decoded_rels.append([class_name[classes[rel[0]]]+'-'+rels_name[rel[2]]+'-'+class_name[classes[rel[1]]]])
    return  all_decoded_rels

def inrease_rect(boxes, factor):
    #increase the size of BBox(x1,y1,x2,y2) to include nearest object
    boxes = center_size(boxes)
    boxes[:, 2:] *= factor
    boxes = point_form(boxes)
    boxes[:, :2] = boxes[:, :2].clip(min=0)
    boxes[:, 2:] = boxes[:, 2:].clip(max=1024)

    return  boxes

def load_unscaled(fn):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    im_scale = 1024.0 / max(image_unpadded.size)

    image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                  resample=Image.BICUBIC)
    return image

def lengths(x):
    if isinstance(x,list):
        yield len(x)
        for y in x:
            yield from lengths(y)

def get_params(conf, detector, lr, show_name =True):
    '''get all trainable param with reduced and non reduced lr'''
    # todo clean code , temp for expriment
    if conf.train_obj_roi or conf.train_detector or conf.reduce_lr_obj_enc:
        # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
        # stabilize the models.
        fc_params = []
        non_fc_params = []
        n_fc_params = []
        n_non_fc_params = []
        for n, param in detector.named_parameters():
            if param.requires_grad:
                if n.startswith(('detector','roi_extractor')) and conf.train_detector:
                    if conf.reduce_lr:
                        fc_params.append(param)
                        n_fc_params.append(n)
                    else:
                        non_fc_params.append(param)
                        n_non_fc_params.append(n)

                if (n.startswith(('roi_fmap_obj', 'context.decoder_lin'))) and conf.train_obj_roi: #'roi_fmap' todo use normal lr for edge
                    if conf.reduce_lr and conf.dataset=='vg':
                        fc_params.append(param)
                        n_fc_params.append(n)
                    else:
                        non_fc_params.append(param)
                        n_non_fc_params.append(n)

                if n.startswith(('context.obj_ctx_enc', 'context.compress_node', 'context.pos_embed.0', 'context.pos_embed.1')): #todo should i remove for nomalized box?
                        if conf.reduce_lr_obj_enc:
                            fc_params.append(param)
                            n_fc_params.append(n)
                        else:
                            non_fc_params.append(param)
                            n_non_fc_params.append(n)

                if not (n.startswith(('roi_fmap_obj', 'context.decoder_lin', 'context.pos_embed.0', 'context.pos_embed.1', 'context.obj_ctx_enc', 'context.compress_node', 'detector', 'roi_extractor'))):  #for edge, sub_ob_emb, final emb
                    non_fc_params.append(param)
                    n_non_fc_params.append(n)

        params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
        if show_name:
            print('Reduced lr : ',n_fc_params)
            print('Not reduced lr : ')
            [print(name) for name in n_non_fc_params]
    else:
        params = [p for n, p in detector.named_parameters() if p.requires_grad]

    return params


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_fwd_inv_rels(all_comb, rels, val=False):
    def stack_and_cat(to_rels, indices, obj1, obj2):
        """
        concatinate with previous relation if they exists, else make relation for background
        """
        if len(indices) > 0:
            if to_rels is not None:
                rel_temp = np.column_stack((np.full((len(indices)), i), rels[indices]))
                to_rels = np.concatenate([to_rels, rel_temp], axis=0)
            else:
                to_rels = np.column_stack((np.full((len(indices)), i), rels[indices]))
        else:
            if not val:  #for validation and test only gt rels
                if to_rels is not None:
                    rel_temp = np.column_stack(( i, obj1, obj2, 0))
                    to_rels = np.concatenate([to_rels, rel_temp], axis=0)
                else:
                    to_rels = np.column_stack(( i, obj1, obj2, 0))

        return to_rels
        # rearrange relation as per object combination order, format [pos of unq comb, rel number]

    #split to fwd and inv relation and add bg relation
    fwd_rels = None
    inv_rels = None
    if all_comb.shape[1]<3:
        all_comb = np.concatenate((all_comb,np.zeros((all_comb.shape[0],1),dtype=int)),1)
    for i, each_comb in enumerate(all_comb):
        fwd_indices = np.where((rels[:, :2] == each_comb[:2]).all(axis=1))[0]
        fwd_rels = stack_and_cat(fwd_rels, fwd_indices, each_comb[0], each_comb[1])

        inv_indices = np.where((rels[:, :2] == each_comb[:2][[1, 0]]).all(axis=1))[0]
        inv_rels = stack_and_cat(inv_rels, inv_indices, each_comb[1], each_comb[0])

        if len(fwd_indices) >0 or len(inv_indices)>0:
            each_comb[2]=1

    return all_comb, fwd_rels, inv_rels

def diff2d(src, tgt):
    diff_idx = np.ones(tgt.shape[0])
    if src.shape[1]==tgt.shape[1]:
        for i, arr in enumerate(src):
            idx = np.where((tgt==arr).all(axis=1))
            if len(idx)>0:
                diff_idx[idx]=0
    return np.where(diff_idx)


