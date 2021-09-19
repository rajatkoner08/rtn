from torch.utils.data.sampler import Sampler
import pickle
import torch
import os
class CustomSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, vg_mini, o_val):
        self.data_source = data_source
        self.vg_mini = vg_mini
        self.o_val = o_val

    def __iter__(self):

        if self.vg_mini:
            l = self.file_op("batch_list.txt")
        elif self.o_val:
            l = self.file_op("batch_list_orig.txt")
        else:
            l = self.file_op("batch_list_full.txt")

        return iter(l)

    def __len__(self):
        return len(self.data_source)

    def file_op(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as fp:  # Unpickling
                l = pickle.load(fp)
        else:
            l = torch.randperm(len(self.data_source)).tolist()
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump(l, fp)
        return l