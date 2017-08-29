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

class TinyDCGANFirstInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(TinyDCGANFirstInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        ndf = 64
        self.planes = [ndf,
                       ndf * 2,
                       ndf * 4,
                       ndf * 4]
        self.kernels = [3, 3, 3, 3]
        self.strides = [2, 2, 2, 1]
        if input_dims[-1] >= 16:
            self.pads = [1, 1, 0, 0]
        else:
            self.pads = [1, 1, 1, 1]

        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(in_planes,
                                               self.planes[l],
                                               in_height,
                                               in_width,
                                               self.kernels[l],
                                               padding=self.pads[l],
                                               stride=self.strides[l])))
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[l]
            self.convs.append(nn.Conv2d(in_planes,
                                        int(self.planes[l]),
                                        self.kernels[l],
                                        padding=self.pads[l],
                                        stride=self.strides[l]))
            if l < len(self.planes) - 1:
                self.bns.append(nn.BatchNorm2d(int(self.planes[l])))

        print(self.out_dims)
        # self.conv1 = nn.Conv2d(input_dims[0], 32, 3)
        self.input_lin = nn.Linear(prod(self.out_dims[-1]), hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        current = x_t
        for i, conv in enumerate(self.convs):
            current = conv(current)
            if i < len(self.convs) - 1:
                current = self.bns[i](current)
            current = activation(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        new_hidden = activation(current)

        # joined = torch.cat([current, z_prev], 1)
        # new_hidden = F.tanh(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = self.lin_sigma(new_hidden)
        return (mu, sigma)

class TinyDCGANInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(TinyDCGANInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        ndf = 64
        self.planes = [ndf,
                       ndf * 2,
                       ndf * 4,
                       ndf * 4]
        self.kernels = [3, 3, 3, 3]
        self.strides = [2, 2, 2, 1]
        if input_dims[-1] >= 16:
            self.pads = [1, 1, 0, 0]
        else:
            self.pads = [1, 1, 1, 1]

        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(
                in_planes,
                self.planes[l],
                in_height,
                in_width,
                self.kernels[l],
                padding=self.pads[l],
                stride=self.strides[l])))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            # confusingly this is actually offset by 1
            in_planes, in_height, in_width = self.out_dims[l]

            self.convs.append(nn.Conv2d(in_planes,
                                        self.planes[l],
                                        self.kernels[l],
                                        padding=self.pads[l],
                                        stride=self.strides[l]))

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 3)

        self.input_lin = nn.Linear(prod(self.out_dims[-1]), hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 3, hidden_dim)
        self.merged_lin = nn.Linear(hidden_dim, hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t, prior):
        current = x_t
        for conv in self.convs:
            current = conv(current)
            current = activation(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        current = activation(current)

        joined = torch.cat([current, *prior], 1)
        new_hidden = activation(self.joint_lin(joined))
        new_hidden = activation(self.merged_lin(new_hidden))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = self.lin_sigma(new_hidden)
        return (mu, sigma)

class TinyDCGANGenerator(nn.Module):
    def __init__(self, hidden_dim, output_dims):
        super(TinyDCGANGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        # self.planes[0] will be set automatically so that it is
        # similar to but larger than hidden_dim
        # self.planes = [None, 64, 64, 128, output_dims[0] * 2]
        # self.kernels = [None, 3, 3, 3, 3]

        # DCGAN numbers
        ngf = 64
        self.planes = [None,
                       ngf * 4,
                       ngf * 2,
                       ngf,
                       output_dims[0] * 2]
        self.kernels = [None, 3, 3, 3, 3]
        if output_dims[-1] >= 16:
            self.pads = [None, 0, 1, 1, 1]
        else:
            self.pads = [None, 1, 1, 1, 1]
        self.strides = [None, 1, 1, 1, 1]


        self.in_dims = [output_dims[1:]]
        # self.in_dims = [(7,7)]
        for l in range(len(self.planes) - 1, 0, -1):
            in_dim = conv_transpose_in_dim(*self.in_dims[0],
                                 self.kernels[l],
                                 stride=self.strides[l],
                                 padding=self.pads[l])
            # in_dim = list(int(d / 2) for d in in_dim)
            self.in_dims = [in_dim] + self.in_dims

        self.planes[0] = math.ceil(hidden_dim / prod(self.in_dims[0]))
        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, self.planes[0] * prod(self.in_dims[0]))])
        # import pdb; pdb.set_trace()

        # print("Generator: ", self.in_dims)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(1, len(self.planes)):
            in_height, in_width = self.in_dims[l]
            self.convs.append(nn.ConvTranspose2d(self.planes[l-1],
                                        self.planes[l],
                                        self.kernels[l],
                                        stride=self.strides[l],
                                        padding=self.pads[l]))
            if l < len(self.planes) - 1:
                # pdb.set_trace()
                self.bns.append(nn.BatchNorm2d(self.planes[l]))

        self.convs[-1].bias.data.add_(0.5)

    def forward(self, input):
        current = input
        for lin in self.lins:
            current = lin(current)
            current = activation(current)

        current = current.resize(current.size(0),
                                 self.planes[0],
                                 *self.in_dims[0])
        for i, conv in enumerate(self.convs):
            current = activation(current)

            # current = self.upsample(current)
            current = conv(current)
            if i < len(self.convs) - 1:
                current = self.bns[i](current)

        mu = F.sigmoid(current[:, : int(current.size(1) / 2)])
        sigma = Variable(torch.ones(mu.size()).type(dtype) * opt.output_var)
        return (mu, sigma)