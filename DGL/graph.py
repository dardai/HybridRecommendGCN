import dgl
import numpy as np
import torch
import torch.nn as nn

node_feat_dim = 2

start_vertices = np.array([0, 1, 2, 3])
dest_vertices = np.array([1, 3, 1, 2])

u = np.concatenate([start_vertices, dest_vertices])
v = np.concatenate([dest_vertices, start_vertices])

g = dgl.graph((u, v))

g.ndata['feat'] = torch.FloatTensor(np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))

print(g)
print(g.ndata['feat'])

linear = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim)))
print(linear)
print(g.ndata['feat']*linear)
