import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import pdb

from util import *
from params import *

if opt.activation == 'lrelu':
    activation = F.leaky_relu
elif opt.activation == 'tanh':
    activation = F.tanh
elif opt.activation == 'selu':
    activation = F.selu
else:
    raise Exception("Activation was not specified properly.")

class ConvFirstInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(ConvFirstInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.planes = [128, 64, 64, 16, 16]
        self.kernels = [3, 3, 3, 3, 3]

        # self.planes = [32, 16]
        # self.kernels = [3, 3]
        #
        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(in_planes,
                                               self.planes[l],
                                               in_height,
                                               in_width,
                                               self.kernels[l],
                                               padding=1,
                                               stride=2)))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[l]

            self.convs.append(nn.Conv2d(
                in_planes, self.planes[l], self.kernels[l], padding=1,
                stride=2))

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 3)
        self.input_lin = nn.Linear(prod(self.out_dims[-1]), hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        current = x_t
        for conv in self.convs:
            current = conv(current)
            current = activation(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        new_hidden = activation(current)

        # joined = torch.cat([current, z_prev], 1)
        # new_hidden = F.tanh(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class ConvInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(ConvInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.planes = [128, 64, 64, 16, 16]
        self.kernels = [3, 3, 3, 3, 3]

        # self.planes = [32, 16]
        # self.kernels = [3, 3]

        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(
                in_planes,
                self.planes[l],
                in_height,
                in_width,
                self.kernels[l],
                padding=1,
                stride=2)))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            # confusingly this is actually offset by 1
            in_planes, in_height, in_width = self.out_dims[l]

            self.convs.append(nn.Conv2d(
                in_planes, self.planes[l], self.kernels[l], padding=1,
                stride=2))

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 3)

        self.input_lin = nn.Linear(prod(self.out_dims[-1]), hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t, z_prev):
        current = x_t
        for conv in self.convs:
            current = conv(current)
            current = activation(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        current = activation(current)

        joined = torch.cat([current, z_prev], 1)
        new_hidden = activation(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class ConvGenerator(nn.Module):
    def __init__(self, hidden_dim, output_dims):
        super(ConvGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        # self.planes[0] will be set automatically so that it is
        # similar to but larger than hidden_dim
        self.planes = [None, 64, 64, 128, output_dims[0] * 2]
        self.kernels = [None, 3, 3, 3, 3]

        # self.planes = [1, 64, output_dims[0] * 2]
        # self.kernels = [None, 3, 3]


        self.in_dims = [output_dims[1:]]
        for l in range(len(self.planes) - 1, 0, -1):
            in_dim = conv_in_dim(*self.in_dims[0],
                                 self.kernels[l],
                                 padding=1)
            in_dim = list(int(d / 2) for d in in_dim)
            self.in_dims = [in_dim] + self.in_dims

        self.planes[0] = math.ceil(hidden_dim / prod(self.in_dims[0]))
        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, self.planes[0] * prod(self.in_dims[0]))])
        # import pdb; pdb.set_trace()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.convs = nn.ModuleList()

        print(self.in_dims)
        for l in range(1, len(self.planes)):
            in_height, in_width = self.in_dims[l]
            self.convs.append(nn.Conv2d(self.planes[l-1],
                                        self.planes[l],
                                        self.kernels[l],
                                        padding=1))

        self.convs[-1].bias.data.add_(0.5)

    def forward(self, input):
        current = input
        for lin in self.lins:
            current = lin(current)
            current = activation(current)

        current = current.resize(current.size(0),
                                 self.planes[0],
                                 *self.in_dims[0])
        for conv in self.convs:
            current = activation(current)
            current = self.upsample(current)
            current = conv(current)

        # print(current.size())
        mu = F.leaky_relu(current[:, : int(current.size(1) / 2)])
        # sigma = F.sigmoid(current[:, current.size(1) / 2 :]) + 3e-2
        sigma = Variable(torch.ones(mu.size()).type(dtype) * opt.output_var)
        return (mu, sigma)