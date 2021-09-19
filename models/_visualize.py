"""
Visualization script. I used this to create the figures in the paper.

WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
from lib.context_model import TransformerModel
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.helper import diff2d
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE,DATA_PATH
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os,pickle
from functools import reduce
filepath = os.path.dirname(os.path.abspath(__file__))


conf = ModelConfig(file = os.path.join(filepath,'param.txt')) #write all param to file

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                              use_proposals=conf.use_proposals,o_valid = conf.o_valid,
                              filter_non_overlap=conf.mode == 'sgdet', require_overlap = conf.require_overlap_det)
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size, num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus, vg_mini =conf.vg_mini, o_val =conf.o_valid )

#todo remove this constraint if more GPU are there
#torch.cuda.set_device(1)
detector = TransformerModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates, config=conf )
detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])


############################################ HELPER FUNCTIONS ###################################

def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index)) * (255 - pad) + pad)

    return map_index_to_rgb_color


cmap = get_cmap(len(train.ind_to_classes) + 1)


def load_unscaled(fn, use_im_scale=False):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    if use_im_scale:
        im_scale = IM_SCALE/ max(image_unpadded.size)
        image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                      resample=Image.BICUBIC)
        pred_box_scale = 1.0
        gt_box_scale = IM_SCALE / BOX_SCALE
    else:
        im_scale = BOX_SCALE / max(image_unpadded.size)
        pred_box_scale = BOX_SCALE / IM_SCALE
        gt_box_scale = 1.0
        image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                      resample=Image.BICUBIC)
    return image, pred_box_scale, gt_box_scale


font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 12)


def draw_box(draw, boxx, cls_ind, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=2)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=2)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=2)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=2)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    # print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
    #     h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill=(255,255,255,255), font=font) #'black'
    return draw


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator)

    evaluator[conf.mode].print_stats()


def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
    #load image
    theimg, pred_box_scale, gt_box_scale = load_unscaled(val.filenames[batch_num])
    theimg2 = theimg.copy()
    draw1 = ImageDraw.Draw(theimg)
    draw2 = ImageDraw.Draw(theimg2)

    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]
    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, obj_preds, p_o_r_m, g_o_r_m, pred_edge,
            true_edge) in enumerate(det_res[0]):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num * conf.batch_size + i].copy(),
            'gt_relations': val.relationships[batch_num * conf.batch_size + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num * conf.batch_size + i].copy(),
        }
        # assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)  #as it may contain background also
        no_of_gt_class = len(gt_entry['gt_classes'])  # to get rid of padding
        assert np.max(rels_i) <= no_of_gt_class - 1

        pred_entry = {
            'pred_boxes': (boxes_i * BOX_SCALE / IM_SCALE)[:no_of_gt_class, :],
            'pred_classes': objs_i[:no_of_gt_class],
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i[:no_of_gt_class],
            'rel_scores': pred_scores_i,  # hack for now.
            'obj_preds': obj_preds[:no_of_gt_class],
            'p_o_r_m': p_o_r_m,
            'g_o_r_m': g_o_r_m,
            'pred_edge': pred_edge,
            'true_edge': true_edge,
        }
        assert len(pred_entry['pred_classes']) == len(gt_entry['gt_classes'])

        pred_to_gt, pred_5ples, rel_scores = evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
    # SET RECALL THRESHOLD HERE
    pred_to_gt = pred_to_gt[:20]
    pred_5ples = pred_5ples[:20]

    # Get a list of objects that match, and GT objects that dont
    if conf.mode=='sgdet':
        objs_match = (bbox_overlaps(pred_entry['pred_boxes'], gt_entry['gt_boxes']) >= 0.5) & (
            objs_i[:, None] == gt_entry['gt_classes'][None])
    else:
        objs_match = (objs_i[:, None] == gt_entry['gt_classes'][None])

    objs_matched = objs_match.any(1)

    has_seen = defaultdict(int)
    has_seen_gt = defaultdict(int)
    pred_ind2name = {}
    gt_ind2name = {}
    edges = {}
    missededges = {}
    badedges = {}

    if val.filenames[batch_num].startswith('2343676'):
        import ipdb
        ipdb.set_trace()

    def query_pred(pred_ind):
        if pred_ind not in pred_ind2name:
            has_seen[objs_i[pred_ind]] += 1
            pred_ind2name[pred_ind] = '{}-{}'.format(train.ind_to_classes[objs_i[pred_ind]],
                                                     has_seen[objs_i[pred_ind]])
        return pred_ind2name[pred_ind]

    def query_gt(gt_ind):
        gt_cls = gt_entry['gt_classes'][gt_ind]
        if gt_ind not in gt_ind2name:
            has_seen_gt[gt_cls] += 1
            gt_ind2name[gt_ind] = '{}-GT{}'.format(train.ind_to_classes[gt_cls], has_seen_gt[gt_cls])
        return gt_ind2name[gt_ind]

    matching_pred5ples = pred_5ples[np.array([len(x) > 0 for x in pred_to_gt])]
    for fiveple in matching_pred5ples:
        head_name = query_pred(fiveple[0])
        tail_name = query_pred(fiveple[1])

        edges[(head_name, tail_name)] = train.ind_to_predicates[fiveple[4]]   #it stores the correct pred edges

    gt_5ples = np.column_stack((gt_entry['gt_relations'][:, :2],
                                gt_entry['gt_classes'][gt_entry['gt_relations'][:, 0]],
                                gt_entry['gt_classes'][gt_entry['gt_relations'][:, 1]],
                                gt_entry['gt_relations'][:, 2],
                                ))
    has_match = reduce(np.union1d, pred_to_gt)
    for gt in gt_5ples[np.setdiff1d(np.arange(gt_5ples.shape[0]), has_match)]:
        # Head and tail
        namez = []
        for i in range(2):
            matching_obj = np.where(objs_match[:, gt[i]])[0]
            if matching_obj.size > 0:
                name = query_pred(matching_obj[0])
            else:
                name = query_gt(gt[i])
            namez.append(name)

        missededges[tuple(namez)] = train.ind_to_predicates[gt[4]]

    for fiveple in pred_5ples[diff2d(matching_pred5ples, pred_5ples )]:

        if fiveple[0] in pred_ind2name:
            if fiveple[1] in pred_ind2name:
                badedges[(pred_ind2name[fiveple[0]], pred_ind2name[fiveple[1]])] = train.ind_to_predicates[fiveple[4]]

    # Fix the names
    two_boxes = []
    for i, pred_ind in enumerate(pred_ind2name.keys()):
        # if i>1:
        #     obj_comb = np.expand_dims([0,1],0)
        #     two_boxes = np.asarray(two_boxes)
        #     union_box = np.concatenate((np.min((two_boxes[:, 1:3][obj_comb[:, 0]], two_boxes[:, 1:3][obj_comb[:, 1]]), axis=0),
        #                 np.max((two_boxes[:, 3:5][obj_comb[:, 0]], two_boxes[:, 3:5][obj_comb[:, 1]]), axis=0)),1)
        #     draw2 = draw_box(draw2, union_box[0,:],
        #                      cls_ind=objs_i[pred_ind],
        #                      text_str='edge')
        #     break
        two_boxes.append(np.insert(pred_entry['pred_boxes'][pred_ind], 0, 0))
        draw2 = draw_box(draw2, pred_entry['pred_boxes'][pred_ind],
                         cls_ind=objs_i[pred_ind],
                         text_str=pred_ind2name[pred_ind])
    for gt_ind in gt_ind2name.keys():
        draw1 = draw_box(draw1, gt_entry['gt_boxes'][gt_ind],
                         cls_ind=gt_entry['gt_classes'][gt_ind],
                         text_str=gt_ind2name[gt_ind])

    recall = int(100 * len(reduce(np.union1d, pred_to_gt)) / gt_entry['gt_relations'].shape[0])

    id = '{}-{}'.format(val.filenames[batch_num].split('/')[-1][:-4], recall)
    pathname = os.path.join(DATA_PATH,'qualitative', id)
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    theimg.save(os.path.join(pathname, 'img.jpg'), quality=100, subsampling=0)
    theimg2.save(os.path.join(pathname, 'imgbox.jpg'), quality=100, subsampling=0)
    if val.filenames[batch_num].split('/')[-1][:-4]=='2343727':
        print('test')
    with open(os.path.join(pathname, 'shit.txt'), 'w') as f:
        f.write('Recall : '+str(recall)+'\n')
        f.write('good:\n')
        for (o1, o2), p in edges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        f.write('False Negative:\n')
        for (o1, o2), p in missededges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        f.write('shit:\n')
        for (o1, o2), p in badedges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))


mAp = val_epoch()
