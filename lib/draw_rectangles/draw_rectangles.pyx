######
# Draws rectangles
######

import cython
from config import IM_SCALE
import numpy as np
cimport numpy as np


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def draw_union_boxes(bbox_pairs, union_box, tgt_seq, pooling_size, padding=0):
    """
    Draws union boxes for the image.
    :param box_pairs: [num_pairs, 8]
    :param fmap_size: Size of the original feature map
    :param stride: ratio between fmap size and original img (<1)
    :param pooling_size: resize everything to this size
    :return: [num_pairs, 2, pooling_size, pooling_size arr
    """
    assert padding == 0, "Padding>0 not supported yet"
    return draw_union_boxes_c(bbox_pairs,union_box, tgt_seq, pooling_size)

cdef DTYPE_t minmax(DTYPE_t x):
    return min(max(x, 0), 1)

cdef np.ndarray[DTYPE_t, ndim=4] draw_union_boxes_c(
        np.ndarray[DTYPE_t, ndim=2] box_pairs, np.ndarray[DTYPE_t, ndim=2] union_box, np.ndarray[DTYPE_t, ndim=2] tgt_seq, unsigned int pooling_size):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float. everything has arbitrary ratios
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = box_pairs.shape[0]

    cdef np.ndarray[DTYPE_t, ndim = 4] uboxes = np.zeros(
        (N, 2, pooling_size, pooling_size), dtype=DTYPE)

    cdef DTYPE_t x1_union, y1_union, x2_union, y2_union, w, h, x1_box, y1_box, x2_box, y2_box, y_contrib, x_contrib, un_y_contrib, un_x_contrib
    cdef unsigned int n, i, j, k

    for n in range(N):
        if tgt_seq[n,1]==1.0: #filtering out all paded boxes

            x1_union = union_box[n,0] #min(box_pairs[n, 0], box_pairs[n, 4])
            y1_union = union_box[n,1] #min(box_pairs[n, 1], box_pairs[n, 5])
            x2_union = union_box[n,2] #max(box_pairs[n, 2], box_pairs[n, 6])
            y2_union = union_box[n,3] #max(box_pairs[n, 3], box_pairs[n, 7])

            w = x2_union - x1_union
            h = y2_union - y1_union

            # box1 = box_pairs[n, 0:4]*pooling_size/IM_SCALE
            # box2 = box_pairs[n, 4:8]*pooling_size/IM_SCALE

            for i in range(2):
                # Now everything is in the range [0, pooling_size].
                x1_box = (box_pairs[n, 0+4*i] - x1_union)*pooling_size / w
                y1_box = (box_pairs[n, 1+4*i] - y1_union)*pooling_size / h
                x2_box = (box_pairs[n, 2+4*i] - x1_union)*pooling_size / w
                y2_box = (box_pairs[n, 3+4*i] - y1_union)*pooling_size / h
                # print("{:.3f}, {:.3f}, {:.3f}, {:.3f}".format(x1_box, y1_box, x2_box, y2_box))
                for j in range(pooling_size):
                    y_contrib = minmax(j+1-y1_box)*minmax(y2_box-j)
                    # if i == 0:
                    #     un_y_contrib = minmax(j + 1 - y1_union*pooling_size/IM_SCALE) * minmax(y2_union*pooling_size/IM_SCALE - j)

                    for k in range(pooling_size):
                        x_contrib = minmax(k+1-x1_box)*minmax(x2_box-k)
                        # if i == 0:
                        #     un_x_contrib = minmax(k + 1 - x1_union*pooling_size/IM_SCALE) * minmax(x2_union*pooling_size/IM_SCALE - k)
                        #     uboxes[n,0,j,k] = un_x_contrib * un_y_contrib

                        # print("j {} yc {} k {} xc {}".format(j, y_contrib, k, x_contrib))
                        uboxes[n,i,j,k] = x_contrib*y_contrib
        #print(" shape of uboxes array : ", uboxes.size)
    return uboxes
