# -*-coding:utf-8-*-
import torch
import numpy as np

# Flat
def build_nefe(ne, fe):

    feshape = fe.shape
    batch_size = feshape[0]
    n_channels = feshape[1]
    n_num = feshape[2]

    nefe = []
    for i in range(batch_size):
        ne1 = ne[i, :, :]
        fe1 = fe[i, :, :]

        fe1 = fe1.permute(1, 0).contiguous()
        ne1 = ne1.permute(1, 0).contiguous()
        ne_flat1 = ne1.view(-1).long()
        efe1 = torch.index_select(fe1, dim=0, index=ne_flat1)

        efe1 = efe1.permute(1, 0)
        efe1 = efe1.view(fe1.shape[1], ne1.shape[0], ne1.shape[1])
        efe1 = efe1.permute(0, 2, 1)
        efe1 = efe1.unsqueeze(0)
        nefe.append(efe1)
    nefe = torch.cat(nefe, dim=0)
    return nefe

