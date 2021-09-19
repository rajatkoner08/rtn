import os
import numpy as np
import argparse


from MOT_Constants import DATA_DIR
from MOT_Constants import LOG_DIR

def find_avg(FLAGS):
    dataset_name = FLAGS.datasetName

    # add image info based on seq length , then no of target per frame,
    # if no of target exceeds the limit, then add it in next epoch
    label_path = os.path.join(DATA_DIR, dataset_name, 'labels', 'train')
    boxes = np.load(os.path.join(label_path, 'Boxes.npy'))

    width = boxes[:,4] - boxes[:,2]
    height = boxes[:,5] - boxes[:,3]
    aspect_ratio = width/height
    print('Max width : ',np.max(width))
    print('Max Height',np.max(height))
    print('Min width : ', np.min(width))
    print('Min Height', np.min(height))
    print('Avg width', np.average(width))
    print('Avg height', np.average(height))
    print('Avg ascept ratio', np.average(aspect_ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for network images.')

    parser.add_argument('-l', '--seq_length', action='store', default=4,
            dest='seq_length', type=int)

    parser.add_argument('-d', '--debug', action='store_true', default=False,
            dest='debug')
    parser.add_argument('-b', '--batch', action='store', default=1,
                        dest='batch_size',type=int)
    parser.add_argument('-ds', '--dataset_name', type=str, default='DETRAC',
                        dest='datasetName',help='Dataset name')
    args = parser.parse_args()
    find_avg(args)

