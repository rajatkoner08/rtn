import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# sphinx_gallery_thumbnail_number = 2
from config import VRD_DATA_DIR

#numbering in fomated way
def format_name(gt_classes):
    gt_cls_number = []
    for i, name in enumerate(gt_classes):
        occ = sum(name in s for s in gt_cls_number)
        if occ==0:
            gt_cls_number.append(name)
        else:
            gt_cls_number.append('{}_{}'.format(name, occ))
    return np.asarray(gt_cls_number)

def process_obj_attn_map(attn_map, class_name, src_seq, class_idx, batchsize, filename, good_img, pos = 0):
    top_attn_map = attn_map[2]
    seq =  src_seq.shape[0]//batchsize
    top_attn_map = top_attn_map.view(12, batchsize, seq, seq).permute(1,0,2,3).data.cpu().numpy()
    class_idx = class_idx.view(batchsize,src_seq.shape[0]//batchsize,2)[:,:,1].data.cpu().numpy()
    src_seq = src_seq.view(batchsize,src_seq.shape[0]//batchsize,2)[:,:,1].data.cpu().numpy()
    for i, attn_map in enumerate(top_attn_map):
        if i in good_img:
            imagename = filename[i]
            save_img = os.path.splitext(os.path.basename(imagename))[0] + '_o_o'
            if save_img == '2343729_o_o':
                print('test')
            valid_seq = len(np.where(src_seq[i])[0])
            obj_names= format_name(class_name[class_idx[i,:valid_seq]])
            map = attn_map[pos,:valid_seq,:valid_seq]
            draw_save_attn_map(obj_names,obj_names, np.around(map,decimals=2), save_img)

#save node-node, or edge-edge attn map
def process_edge_attn_map(attn_map, class_name, src_seq, class_idx, batchsize, filename, good_img, obj_comb=None, tgt_seq=None, pos = 0):
    top_attn_map = attn_map[1]
    obj_comb = obj_comb.view(batchsize, obj_comb.shape[0]//batchsize, 3)[:, :, 1:].data.cpu().numpy()
    tgt_seq = tgt_seq.view(batchsize, tgt_seq.shape[0] // batchsize, 2)[:, :, 1].data.cpu().numpy()
    top_attn_map = top_attn_map.view(12, batchsize, tgt_seq.shape[1], tgt_seq.shape[1]).permute(1,0,2,3).data.cpu().numpy()
    class_idx = class_idx.view(batchsize, class_idx.shape[0] // batchsize, 2)[:, :, 1].data.cpu().numpy()
    src_seq = src_seq.view(batchsize, src_seq.shape[0] // batchsize, 2)[:, :, 1].data.cpu().numpy()

    for i, attn_map in enumerate(top_attn_map):
        if i in good_img:
            imagename = filename[i]
            src_valid_seq = len(np.where(src_seq[i])[0])
            tgt_valid_seq = len(np.where(tgt_seq[i])[0])
            obj_names= format_name(class_name[class_idx[i,:src_valid_seq]])
            edge_name = []
            for j in range(0, tgt_valid_seq):
                edge_name.append(obj_names[obj_comb[i, j, 0]] + '-' + obj_names[obj_comb[i, j, 1]])
            obj_names = np.asarray(edge_name)
            #pos = np.unravel_index(attn_map[:, :tgt_valid_seq, :tgt_valid_seq].argmax(),  attn_map[:, :tgt_valid_seq, :tgt_valid_seq].shape)[0]

            map = attn_map[pos,:tgt_valid_seq,:tgt_valid_seq]
            save_img = os.path.splitext(os.path.basename(imagename))[0]+'_e_e'
            # if save_img == '2343398e_e':
            #     print('test')
            draw_save_attn_map(obj_names,obj_names, np.around(map,decimals=2), save_img, edge='ee')

#save node_edge attn map
def process_node_edge_map(attn_map, class_name, src_seq, class_idx, batchsize, filename, tgt_seq, obj_comb, pos = 0):
    top_attn_map = attn_map[1]
    pad_srcseq = src_seq.shape[0] // batchsize
    pad_tgt_seq = tgt_seq.shape[0]//batchsize
    top_attn_map = top_attn_map.view(12, batchsize, pad_tgt_seq, pad_srcseq).permute(1,0,2,3).data.cpu().numpy()
    class_idx = class_idx.view(batchsize, src_seq.shape[0] // batchsize, 2)[:, :, 1].data.cpu().numpy()
    src_seq = src_seq.view(batchsize, src_seq.shape[0] // batchsize, 2)[:, :, 1].data.cpu().numpy()
    obj_comb = obj_comb.view(batchsize, tgt_seq.shape[0] // batchsize, 3)[:,:,1:].data.cpu().numpy()
    tgt_seq = tgt_seq.view(batchsize, tgt_seq.shape[0] // batchsize, 2)[:, :, 1].data.cpu().numpy()
    good_image = []
    for i, attn_map in enumerate(top_attn_map):
        if np.max(top_attn_map)>=0.0:
            good_image.append(i)
            imagename = filename[i]
            valid_seq = len(np.where(src_seq[i])[0])
            valid_tgt_seq = len(np.where(tgt_seq[i])[0])
            obj_names = format_name(class_name[class_idx[i, :valid_seq]])
            edge_name = []
            for j in range(0,valid_tgt_seq):
                edge_name.append(obj_names[obj_comb[i,j,0]]+'-'+obj_names[obj_comb[i,j,1]])
            edge_name =np.asarray(edge_name)
            if os.path.splitext(os.path.basename(imagename))[0] in ('2343408','2340675','2340764'):
                #print('test')
                for j, attn_head_map in enumerate(attn_map):
                    #pos = np.unravel_index(attn_map[:,:valid_tgt_seq, :valid_seq].argmax(), attn_map[:, :valid_tgt_seq, :valid_seq].shape)[0]
                    map = attn_head_map[:valid_tgt_seq, :valid_seq]


                    draw_save_attn_map(edge_name, obj_names, np.around(map, decimals=2),
                                       os.path.splitext(os.path.basename(imagename))[0] + '_n_e'+str(j), edge='en')

        return good_image


def draw_save_attn_map(query, key, values, im_name, edge=None):
    if edge =='en':
        pathname = os.path.join(VRD_DATA_DIR, 'attention_n_e')
    else:
        return
        pathname = os.path.join(VRD_DATA_DIR, 'attention')
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    fig, ax = plt.subplots()
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(key)))
    ax.set_yticks(np.arange(len(query)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(key)
    ax.set_yticklabels(query)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(query)):
        for j in range(len(key)):
            text = ax.text(j, i, values[i, j],
                           ha="center", va="center", color="w")

    if edge=='en':
        ax.set_title("Edge-Node Attention")
    elif edge=='ee':
        ax.set_title("Edge-Edge Attention")
    else:
        ax.set_title("Node-Node Attention")
    fig.tight_layout()
    #plt.show()

    fig.savefig(os.path.join(pathname, str(im_name) + '.png'))