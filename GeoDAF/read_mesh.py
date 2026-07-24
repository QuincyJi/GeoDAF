# -*-coding:utf-8-*-

import torch
import trimesh
import numpy as np
import os


class Mesh:
    def __init__(self, file=None):
        super(Mesh, self).__init__()

        mesh = trimesh.load(file)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.fn = mesh.face_normals.astype(np.float32)
        self.face_adj = mesh.face_adjacency
        self.ne_1 = self.get_neighbors1(self.face_adj)
        self.ne_0 = np.arange(self.faces.shape[0]).reshape(-1, 1)
        self.ne = np.hstack((self.ne_0, self.ne_1))
        self.ne = self.ne.astype(int)
        self.face_areas = mesh.area_faces.reshape(-1, 1)
        self.face_center = mesh.triangles_center

        # The cross product of two edges of each triangle.
        self.cross = mesh.triangles_cross.astype(np.float32)
        self.trangles = mesh.triangles
        self.corners = self.get_corners(self.face_center, self.trangles)

        self.features = self.fn.astype(np.float32)
        self.len = self.faces.shape[0]


    def get_neighbors1(self, adj):
        ne_1 = np.zeros((adj.max() + 1, 3)).astype(int)
        ne_list = np.zeros((adj.max() + 1)).astype(int)

        for i in range(adj.shape[0]):
            edge = adj[i]
            index_1 = edge[0]
            index_2 = edge[1]
            ne_1[index_1][ne_list[index_1]] = edge[1]
            ne_list[index_1] += 1
            ne_1[index_2][ne_list[index_2]] = edge[0]
            ne_list[index_2] += 1
        # 边缘用中心点填充
        indices = [i for i, x in enumerate(ne_list) if x < 3]
        for i in range(indices.__len__()):
            index_1 = indices[i]
            ne_1[index_1][ne_list[index_1]] = index_1
            ne_list[index_1] += 1

        return ne_1

    def get_corners(self, center, triangles):

        corners = np.zeros((len(triangles), 3))

        for j in range(3):
            change = triangles[0:, j, 0:] - center
            distances = np.sqrt(np.sum(change ** 2, axis=1))
            corners[:, j] = distances
        return corners