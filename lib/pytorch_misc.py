"""
Miscellaneous functions that might be useful for pytorch
"""

import h5py
import numpy as np
import torch
from torch.autograd import Variable
import os
import dill as pkl
from itertools import tee
from torch import nn
from lib.helper import lengths
from lib.fpn.box_utils import bbox_overlaps, center_size, relative_size



def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    #############################
    # print("Network layer :")
    # for i in own_state:
    #     print(i+" : "+str(own_state[i].shape))
    # print("Checkpoint layer :")
    # for i in state_dict:
    #     print(i + " : " + str(state_dict[i].shape))
    # import sys
    # sys.exit()
    #####################################

    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_ranking(predictions, labels, num_guesses=5):
    """
    Given a matrix of predictions and labels for the correct ones, get the number of guesses
    required to get the prediction right per example.
    :param predictions: [batch_size, range_size] predictions
    :param labels: [batch_size] array of labels
    :param num_guesses: Number of guesses to return
    :return:
    """
    assert labels.size(0) == predictions.size(0)
    assert labels.dim() == 1
    assert predictions.dim() == 2

    values, full_guesses = predictions.topk(predictions.size(1), dim=1)
    _, ranking = full_guesses.topk(full_guesses.size(1), dim=1, largest=False)
    gt_ranks = torch.gather(ranking.data, 1, labels.data[:, None]).squeeze()

    guesses = full_guesses[:, :num_guesses]
    return gt_ranks, guesses

def cache(f):
    """
    Caches a computation
    """
    def cache_wrapper(fn, *args, **kwargs):
        if os.path.exists(fn):
            with open(fn, 'rb') as file:
                data = pkl.load(file)
        else:
            print("file {} not found, so rebuilding".format(fn))
            data = f(*args, **kwargs)
            with open(fn, 'wb') as file:
                pkl.dump(data, file)
        return data
    return cache_wrapper


class Flattener(nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


def to_variable(f):
    """
    Decorator that pushes all the outputs to a variable
    :param f: 
    :return: 
    """
    def variable_wrapper(*args, **kwargs):
        rez = f(*args, **kwargs)
        if isinstance(rez, tuple):
            return tuple([Variable(x) for x in rez])
        return Variable(rez)
    return variable_wrapper

def arange(base_tensor, n=None):
    new_size = base_tensor.size(0) if n is None else n
    new_vec = base_tensor.new(new_size).long()
    torch.arange(0, new_size, out=new_vec)
    return new_vec


def make_one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    num_classes : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes).zero_()
    target = one_hot.scatter_(1, labels.data.long().cuda(), 1)
    return target

def to_onehot(vec, num_classes, fill=1.0):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(0.0)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec + num_classes*arange_inds] = fill
    return onehot_result

def save_net(fname, net):
    h5f = h5py.File(fname, mode='w')
    for k, v in list(net.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    h5f = h5py.File(fname, mode='r')
    for k, v in list(net.state_dict().items()):
        param = torch.from_numpy(np.asarray(h5f[k]))

        if v.size() != param.size():
            print("On k={} desired size is {} but supplied {}".format(k, v.size(), param.size()))
        else:
            v.copy_(param)


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start+batch_size, len_l))

def batch_map(f, a, batch_size):
    """
    Maps f over the array a in chunks of batch_size.
    :param f: function to be applied. Must take in a block of
            (batch_size, dim_a) and map it to (batch_size, something).
    :param a: Array to be applied over of shape (num_rows, dim_a).
    :param batch_size: size of each array
    :return: Array of size (num_rows, something).
    """
    rez = []
    for s, e in batch_index_iterator(a.size(0), batch_size, skip_end=False):
        print("Calling on {}".format(a[s:e].size()))
        rez.append(f(a[s:e]))

    return torch.cat(rez)


def const_row(fill, l, volatile=False):
    input_tok = Variable(torch.LongTensor([fill] * l),volatile=volatile)
    if torch.cuda.is_available():
        input_tok = input_tok.cuda()
    return input_tok


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def nonintersecting_2d_inds(x):
    """
    Returns np.array([(a,b) for a in range(x) for b in range(x) if a != b]) efficiently
    :param x: Size
    :return: a x*(x-1) array that is [(0,1), (0,2)... (0, x-1), (1,0), (1,2), ..., (x-1, x-2)]
    """
    rs = 1 - np.diag(np.ones(x, dtype=np.int32))
    relations = np.column_stack(np.where(rs))
    return relations



def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v

def gather_nd(x, index):
    """

    :param x: n dimensional tensor [x0, x1, x2, ... x{n-1}, dim]
    :param index: [num, n-1] where each row contains the indices we'll use
    :return: [num, dim]
    """
    nd = x.dim() - 1
    assert nd > 0
    assert index.dim() == 2
    assert index.size(1) == nd
    dim = x.size(-1)

    sel_inds = index[:,nd-1].clone()
    mult_factor = x.size(nd-1)
    for col in range(nd-2, -1, -1): # [n-2, n-3, ..., 1, 0]
        sel_inds += index[:,col] * mult_factor
        mult_factor *= x.size(col)

    grouped = x.view(-1, dim)[sel_inds]
    return grouped


def enumerate_by_image(im_inds):
    if isinstance(im_inds, np.ndarray):
        im_inds_np = im_inds
    else:
        im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)


# def get_obj_comb_by_batch(fwd_rels, inv_rels):
#
#     def update_idx(rels_np):
#         curr_img_idx = intial_ind = int(rels_np[0][0])
#         temp_max = 1        #as next idx will start from prev +1
#         max_comb_idx = 0
#         for i, val in enumerate(rels_np):
#             max_comb_idx = np.max((max_comb_idx,val[1]))
#             if val[0] != curr_img_idx:
#                 curr_img_idx = val[0]
#                 temp_max += max_comb_idx
#                 max_comb_idx = 0
#             if val[0] != intial_ind:
#                 rels_np[i, 1] += temp_max
#         return rels_np
#
#     fwd_rels_np = fwd_rels.cpu().numpy()
#     inv_rels_np = inv_rels.cpu().numpy()
#
#     fwd_rels = torch.from_numpy(update_idx(fwd_rels_np))
#     inv_rels = torch.from_numpy(update_idx(inv_rels_np))
#
#     return fwd_rels, inv_rels


def get_obj_comb_by_batch(obj_comb, im_inds, fwd_rels, inv_rels, attributes):

    fwd_rel_return = []
    inv_rels_return = []
    #convert both fwd and inv rels to numpy
    if fwd_rels is None:
        fwd_rel_return = None
    elif isinstance(fwd_rels, np.ndarray):
        fwd_rels_np = fwd_rels
    else:
        fwd_rels_np = fwd_rels.cpu().numpy()

    if inv_rels is None:
        inv_rels_return = None
    elif isinstance(inv_rels, np.ndarray):
        inv_rels_np = inv_rels
    else:
        inv_rels_np = inv_rels.cpu().numpy()

    if not attributes is None:
        attributes_np = attributes.cpu().numpy()

    im_offset = {}
    for i, s, e in enumerate_by_image(im_inds):  # increment image number
        im_offset[i] = s

    pred_offset = {}
    for i, s, e in enumerate_by_image(obj_comb[:, 0]):     #increment edge number
        pred_offset[i] = s
        obj_comb[s:e, 1:3] += im_offset[i]                   #incriment object number as per batch

    if not attributes is None:
        for i, attr in enumerate(attributes_np):            #increment image number 4 attributes
            attributes_np[i,1] += im_offset[attributes_np[i,0]]
        attributes_return = torch.from_numpy(attributes_np).cuda()
    else:
        attributes_return = None

    if fwd_rel_return  is not None:
        for i, pred in enumerate(fwd_rels_np):
            fwd_rels_np[i][1] += pred_offset[pred[0]]      #increment edge number
            fwd_rels_np[i][2:4] += im_offset[pred[0]]       #increment image number
        fwd_rel_return = torch.from_numpy(fwd_rels_np).cuda()

    if inv_rels_return is not None:
        for i, pred in enumerate(inv_rels_np):
            inv_rels_np[i][1] += pred_offset[pred[0]]      #increment edge number
            inv_rels_np[i][2:4] += im_offset[pred[0]]       #increment image number
        inv_rels_return = torch.from_numpy(inv_rels_np).cuda()

    return obj_comb, fwd_rel_return, inv_rels_return, attributes_return


def diagonal_inds(tensor):
    """
    Returns the indices required to go along first 2 dims of tensor in diag fashion
    :param tensor: thing
    :return: 
    """
    assert tensor.dim() >= 2
    assert tensor.size(0) == tensor.size(1)
    size = tensor.size(0)
    arange_inds = tensor.new(size).long()
    torch.arange(0, tensor.size(0), out=arange_inds)
    return (size+1)*arange_inds

def enumerate_imsize(im_sizes):
    s = 0
    for i, (h, w, scale, num_anchors) in enumerate(im_sizes):
        na = int(num_anchors)
        e = s + na
        yield i, s, e, h, w, scale, na

        s = e

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def unravel_index(index, dims):
    unraveled = []
    index_cp = index.clone()
    for d in dims[::-1]:
        unraveled.append(index_cp % d)
        index_cp /= d
    return torch.cat([x[:,None] for x in unraveled[::-1]], 1)

def de_chunkize(tensor, chunks):
    s = 0
    for c in chunks:
        yield tensor[s:(s+c)]
        s = s+c

def random_choose(tensor, num):
    "randomly choose indices"
    num_choose = min(tensor.size(0), num)
    if num_choose == tensor.size(0):
        return tensor

    # Gotta do this in numpy because of https://github.com/pytorch/pytorch/issues/1868
    rand_idx = np.random.choice(tensor.size(0), size=num, replace=False)
    rand_idx = torch.LongTensor(rand_idx).cuda(tensor.get_device())
    chosen = tensor[rand_idx].contiguous()

    # rand_values = tensor.new(tensor.size(0)).float().normal_()
    # _, idx = torch.sort(rand_values)
    #
    # chosen = tensor[idx[:num]].contiguous()
    return chosen


def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """

    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def right_shift_packed_sequence_inds(lengths):
    """
    :param lengths: e.g. [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    :return: perm indices for the old stuff (TxB) to shift it right 1 slot so as to accomodate
             BOS toks
             
             visual example: of lengths = [4,3,1,1]
    before:
    
        a (0)  b (4)  c (7) d (8)
        a (1)  b (5)
        a (2)  b (6)
        a (3)
        
    after:
    
        bos a (0)  b (4)  c (7)
        bos a (1)
        bos a (2)
        bos              
    """
    cur_ind = 0
    inds = []
    for (l1, l2) in zip(lengths[:-1], lengths[1:]):
        for i in range(l2):
            inds.append(cur_ind + i)
        cur_ind += l1
    return inds

def clip_grad_norm(named_parameters, max_norm, clip=False, verbose=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        # for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
        #     print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
        # print('-------------------------------', flush=True)

    return total_norm

def update_lr(optimizer, lr=1e-4):
    print("------ Learning rate -> {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lb_pad(base_list, pad_value=0, pad_type = 'buttom', pad_len=None):
    if len(base_list.shape)==1:
        if pad_type=='buttom':
           return np.concatenate((base_list, np.zeros((pad_len), dtype=int)))
        if pad_type == 'left':
            return np.column_stack((np.full(len(base_list),pad_value),base_list ))
    else:
        if pad_type == 'buttom':
            if pad_len == 0:
                return  base_list
            return np.pad(base_list, ((0, pad_len), (0, 0)), 'constant', constant_values=pad_value)
        if pad_type == 'left':
            return np.pad(base_list, ((0, 0), (1, 0)), 'constant', constant_values=pad_value)

def append_and_pad(batch_size, im_inds,  **kwargs):

    #apend img index and pad in same size
    result = []
    obj_com = []
    for key in kwargs:
        new_list = []
        list_all = kwargs[key]
        l_wdout_none = [ l for l in list_all if l is not None] #handeling non is needed for on gt rels
        if len(l_wdout_none)==0:
            result.append(None)
        else:
            max_len = max( len(l) for l in list_all if l is not None)
            for i, each_list in enumerate(list_all):
                if each_list is not None:
                    padded_list = each_list
                    if key not in ('fwd_rel', 'inv_rel'):  # for rels no need to pad buttom
                        padded_list = lb_pad(padded_list, pad_type='buttom', pad_len=max_len - each_list.shape[0])
                    padded_list = lb_pad(padded_list, pad_type='left',pad_value=i)
                    new_list.append(padded_list)

            result.append(np.vstack(new_list))
        # else:
        #     result.append(np.asarray(new_list).reshape(-1))
    #todo remove this as its handeled later in context model
    # offset = {}
    # for i, s, e in enumerate_by_image(result[3][:,0]):  # increment number of obj present in a img
    #     offset[i] = s
    # for i, im in enumerate(result[0].copy()):
    #     im[1:] += offset[im[0]]
    #     obj_com.append(im)

    obj_com_pos = torch.LongTensor(result[0]).cuda()
    fwd_rel = torch.zeros((0,0,0)).type(torch.LongTensor).cuda() if result[1] is None else torch.LongTensor(result[1]).cuda()
    inv_rel = torch.zeros((0,0,0)).type(torch.LongTensor).cuda() if result[2] is None else torch.LongTensor(result[2]).cuda()
    src_seq = torch.LongTensor(result[3]).cuda()
    tgt_seq = torch.LongTensor(result[4]).cuda()
    #obj_com = torch.LongTensor(obj_com).cuda()

    return fwd_rel, inv_rel, src_seq, tgt_seq, obj_com_pos

def get_normalized_rois(rois_normalized, data_height, data_width, spatial_scale):
    #todo check if -1 is ok with get combined feats
    height = (data_height -1) / spatial_scale
    width = (data_width - 1) / spatial_scale

    rois_normalized[:,0] /= width
    rois_normalized[:,1] /= height
    rois_normalized[:,2] /= width
    rois_normalized[:,3] /= height

    return rois_normalized

def get_combined_feats(visual_feats, obj_word_emb, bboxes, compress_feats, normalized_pos=False, pos_embed=None, obj_dist =None, obj_comb = None, edge=False, union=False, spatial_box = False,gap=None):
    #combine vis_feats+obj_class+position for both node and predicate
    if spatial_box:
        pos_feats = torch.zeros((0,0)).cuda()
    elif normalized_pos:
            pos_feats = bboxes  #get_normalized_rois(bboxes, data_height, data_width, spatial_scale=1)
    else:
            pos_feats = pos_embed(Variable(center_size(bboxes)))

    if edge and union:  #for edge and union boxes concat both class embedding
        word_emb_master = torch.cat((obj_word_emb[obj_comb[:, 1]], obj_word_emb[obj_comb[:, 2]]), 1)
    else:
        word_emb_master = obj_word_emb

    if obj_dist is None:
        feats_pre_rep = torch.cat((visual_feats, word_emb_master, pos_feats), 1)
    else:
        feats_pre_rep = torch.cat((visual_feats, word_emb_master, pos_feats, obj_dist), 1)

    if gap is not None and not edge:
        feats_pre_rep = torch.cat((feats_pre_rep, gap), 1)

    # now apply linear projection layer
    fc_feats = compress_feats(feats_pre_rep)

    if not union and edge:  #for edge coming from two nodes
        return fc_feats[obj_comb[:, 1]] * fc_feats[obj_comb[:, 2]]
    else:
        return fc_feats

    #     if edge:
    #         first_obj = torch.cat((visual_feats[obj_comb[:, 1]], obj_word_emb[obj_comb[:, 1]]), 1)
    #         second_obj = torch.cat((visual_feats[obj_comb[:, 2]], obj_word_emb[obj_comb[:, 2]]), 1)
    #         feats_pre_rep = torch.cat((first_obj * second_obj, pos_feats), 1)
    #     else:
    #         feats_pre_rep = torch.cat((visual_feats, obj_word_emb, pos_feats), 1)
    # else:  #todo implement relative box
    #     feats_pre_rep = torch.cat((visual_feats, obj_word_emb, pos_feats, obj_dist), 1)
    #
    # return compress_feats(feats_pre_rep)


def get_feats_size(pooling_dim, use_word_emb, use_normalized_roi, use_union, spatial_box = False, word_dim = 200, num_classes = 151, use_gap=False, gap_dim=512):

    node_dim = 0
    edge_dim = 0
    if spatial_box:
        pos_dim = 0
    elif use_normalized_roi:
        pos_dim = 5
    else:
        pos_dim = 128

    if use_union:
        if use_word_emb:
                edge_dim = pooling_dim + 2 * word_dim + pos_dim #
        else:
                edge_dim = pooling_dim + 2 * num_classes + pos_dim

    if use_word_emb:
        node_dim = pooling_dim + word_dim + pos_dim
    else:
            node_dim = pooling_dim+num_classes+pos_dim

    if edge_dim==0:
        edge_dim = node_dim

    if use_gap:
        node_dim+=gap_dim

    return node_dim, edge_dim

def init_weights(m, use_bias = None):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if use_bias is not None:
            m.bias.data.fill_(use_bias)
        #print(m.weight)

def get_index_pos(seq):
    obj_index = seq.new_zeros(seq.size())
    for i, s, e in enumerate_by_image(torch.nonzero(seq)[:, 0]):
        obj_index[i, :e - s] = torch.arange(1, ((e - s) + 1))
    return obj_index



