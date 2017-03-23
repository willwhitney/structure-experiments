import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

from util import *

def conv_out_dim(in_planes, out_planes, in_height, in_width, kernel_size,
                 stride=1, padding=0, dilation=1):
    dilated_kernel = dilation * (kernel_size - 1)
    out_height = math.floor(
        (in_height + 2 * padding - dilated_kernel - 1) / stride + 1)
    out_width = math.floor(
        (in_width + 2 * padding - dilated_kernel - 1) / stride + 1)
    return out_planes, out_height, out_width

def conv_in_dim(out_height, out_width, kernel_size,
                 stride=1, padding=0, dilation=1):
    dilated_kernel = dilation * (kernel_size - 1)
    # (out_height - 1) * stride = in_height + 2 * padding - dilated_kernel - 1
    in_height = math.ceil(
        (out_height - 1) * stride - 2 * padding + dilated_kernel + 1)
    in_width = math.ceil(
        (out_width - 1) * stride - 2 * padding + dilated_kernel + 1)
    return in_height, in_width

eps = 1e-2
class Transition(nn.Module):
    def __init__(self, hidden_dim):
        super(Transition, self).__init__()
        self.dim = hidden_dim
        self.l1 = nn.Linear(self.dim, self.dim)
        self.lin_mu = nn.Linear(self.dim, self.dim)
        self.lin_sigma = nn.Linear(self.dim, self.dim)


    def forward(self, input):
        hidden = F.tanh(self.l1(input))
        mu = 10 * F.tanh(self.lin_mu(hidden) / 10)
        # sigma = Variable(torch.ones(mu.size()).type(dtype) / 2)
        sigma = F.sigmoid(self.lin_sigma(hidden)) + eps
        # print(sigma.mean().data[0])
        return (mu, sigma)

class Generator(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)
                                     for _ in range(2)])
        self.lin_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
            current = F.tanh(current)

        mu_preactivation = self.lin_mu(current)
        mu = F.sigmoid(mu_preactivation) + 0.1 * mu_preactivation
        # mu = F.leaky_relu(mu_preactivation)

        # sigma = F.sigmoid(self.lin_sigma(current)) + 3e-2
        sigma = Variable(torch.ones(mu.size()).type_as(mu.data) / 50)
        return (mu, sigma)

class ConvolutionalGenerator(nn.Module):
    def __init__(self, hidden_dim, output_dims):
        super(ConvolutionalGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        self.planes = [1, 64, 64, 64, output_dims[0] * 2]
        self.kernels = [None, 3, 3, 3, 3]


        self.in_dims = [output_dims[1:]]
        for l in range(len(self.planes)-1, 0, -1):
            # l = l_dumb - 1
            in_dim = conv_in_dim(*self.in_dims[0],
                                 self.kernels[l],
                                 padding=1)
            self.in_dims = [in_dim] + self.in_dims

        self.lins = nn.ModuleList([
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, prod(self.in_dims[0]))])
        self.convs = nn.ModuleList()

        print(self.in_dims)
        for l in range(1, len(self.planes)):
            in_height, in_width = self.in_dims[l]
            self.convs.append(nn.Conv2d(self.planes[l-1],
                                        self.planes[l],
                                        self.kernels[l],
                                        padding=1))


    def forward(self, input):
        current = input
        for lin in self.lins:
            current = lin(current)
            current = F.tanh(current)

        current = current.resize(current.size(0), 1, *self.in_dims[0])
        for conv in self.convs:
            current = F.tanh(current)
            current = conv(current)

        # print(current.size())
        mu = F.leaky_relu(current[:, : int(current.size(1) / 2)])
        # sigma = F.sigmoid(current[:, current.size(1) / 2 :]) + 3e-2
        sigma = Variable(torch.ones(mu.size()).type_as(mu.data) / 50)
        return (mu, sigma)

class Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Inference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(2)])

        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t, z_prev):
        embedded = F.tanh(self.input_lin(x_t))
        joined = torch.cat([embedded, z_prev], 1)
        new_hidden = F.tanh(self.joint_lin(joined))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)

        mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class ConvolutionalInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(ConvolutionalInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.planes = [32, 16]
        self.kernels = [3, 3]
        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(
                in_planes,
                self.planes[l],
                in_height,
                in_width,
                self.kernels[l],
                padding=0)))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            # confusingly this is actually offset by 1
            in_planes, in_height, in_width = self.out_dims[l]

            self.convs.append(nn.Conv2d(
                in_planes, self.planes[l], self.kernels[l], padding=0))

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
            current = F.tanh(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        current = F.tanh(current)

        joined = torch.cat([current, z_prev], 1)
        new_hidden = F.tanh(self.joint_lin(joined))

        mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class FirstInference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FirstInference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(2)])

        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        new_hidden = F.tanh(self.input_lin(x_t))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)

        mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class ConvolutionalFirstInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(ConvolutionalFirstInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        self.planes = [32, 16]
        self.kernels = [3, 3]
        self.out_dims = [input_dims]
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[-1]
            self.out_dims.append(tuple(conv_out_dim(in_planes,
                                               self.planes[l],
                                               in_height,
                                               in_width,
                                               self.kernels[l],
                                               padding=0)))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            # confusingly this is actually offset by 1
            in_planes, in_height, in_width = self.out_dims[l]

            self.convs.append(nn.Conv2d(
                in_planes, self.planes[l], self.kernels[l], padding=0))

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
            current = F.tanh(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        new_hidden = F.tanh(current)

        # joined = torch.cat([current, z_prev], 1)
        # new_hidden = F.tanh(self.joint_lin(joined))

        mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class GaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, sigma_q) = q
        (mu_p, sigma_p) = p

        a = torch.sum(torch.log(sigma_p), 1) - torch.sum(torch.log(sigma_q), 1)
        b = torch.sum(sigma_q / sigma_p, 1)

        mu_diff = mu_p - mu_q
        c = torch.sum(torch.pow(mu_diff, 2) / sigma_p, 1)

        D = mu_q.size(1)
        divergences = torch.mul(a + b + c - D, 0.5)
        return divergences.mean()

# class GaussianKLD1(nn.Module):
#     def forward(self, q, p):
#         (mu_q, sigma_q) = q
#         (mu_p, sigma_p) = p
#         a1 = torch.sum(torch.pow(sigma_p, -1) * sigma_q, 1)
#         b1 = torch.sum((mu_p - mu_q) * torch.pow(sigma_p, -1) * (mu_p - mu_q), 1)
#         c1 = - mu_q.size(1)
#         d1 = torch.log(torch.prod(sigma_p, 1) / torch.prod(sigma_q, 1))
#         return 0.5 * (a1 + b1 + c1 + d1)

class GaussianLL(nn.Module):
    def forward(self, p, target):
        (mu, sigma) = p

        # sigma = Variable(torch.ones(sigma.size()).type_as(sigma.data) / 10)

        a = torch.sum(torch.log(sigma), 1)
        diff = (target - mu)
        b = torch.sum(torch.pow(diff, 2) / sigma, 1)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()
