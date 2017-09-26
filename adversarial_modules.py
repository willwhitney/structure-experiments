import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import pdb

# from util import *
from params import opt, dtype

if opt.activation == 'lrelu':
    activation = F.leaky_relu
elif opt.activation == 'tanh':
    activation = F.tanh
elif opt.activation == 'selu':
    activation = F.selu
else:
    raise Exception("Activation was not specified properly.")

class Adversary(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=2):
        super().__init__()
        self.layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        current = x
        for l in range(len(self.layers) - 1):
            layer = self.layers[l]
            current = layer(current)
            current = activation(current)
        current = F.sigmoid(self.layers[-1](current))
        return current
