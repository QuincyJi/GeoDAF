# -*-coding:utf-8-*-

import torch
import torch.utils.data as data

from typing import Any, Callable, List, Optional, Tuple, Union
import os.path as osp
import glob
import os
import numpy as np
import pickle
import random

from read_mesh import Mesh
from meshconv_util import pad, pad_sequence, pad_ne


class BaseDataset(data.Dataset):

    def __init__(self, args,
            path: str,
            target_length: int,
            train: bool = True,

    ):
        super(BaseDataset, self).__init__()
        self.args = args
        self.target_length = target_length
        self.path = path
        self.train = train
        if self.train:
            self.dataset = 'train'
        else:
            self.dataset = 'test'

        self.paths, self.label = self.get_path()

    def get_path(self):
        categories = glob.glob(osp.join(self.path, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])
        PATH = []
        label = []
        for target, category in enumerate(categories):
            folder = osp.join(self.path, category, self.dataset)
            paths = glob.glob(folder + '/' + '*.obj')
            PATH.extend(paths)
            for path in paths:
                label.append(target)

        return PATH, label

    def __getitem__(self, index):

        path = self.paths[index]
        label = self.label[index]
        exist, save_path = self.path_InOrNot(path)
        if exist:
            mesh = load_pickle(save_path)
        else:
            mesh = Mesh(file=path)
            mesh.features, index = pad_sequence(mesh.features, self.target_length, 'random')
            mesh.ne = pad_ne(mesh.ne, self.target_length, index)
            mesh.features = mesh.features.T

            folder_path = os.path.dirname(save_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_pickle(mesh, save_path)

        meta = {'feature': mesh.features, 'neighbor': mesh.ne, 'label': label, 'edges_count': mesh.len}
        return meta


    def path_InOrNot(self, path):
        directory = osp.dirname(self.path)
        mid = osp.basename(osp.dirname(osp.dirname(path))) + '/' + osp.basename(osp.dirname(path))
        osp.basename(mid)
        filename = osp.basename(path).split('.')[0]
        foldname = self.args.save_fold
        save_path = osp.join(directory, foldname, mid, str(filename+'.pkl'))
        exist = osp.exists(save_path)

        return exist, save_path

    def __len__(self):
        return len(self.paths)

def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data
def collate_fn(batch):

    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})

    return meta




