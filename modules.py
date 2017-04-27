import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

from util import *
from params import *

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

def conv_transpose_in_dim(out_height, out_width, kernel_size,
                          stride=1, padding=0, dilation=1):
    # dilated_kernel = dilation * (kernel_size - 1)
    dilated_kernel = kernel_size
    in_height = math.ceil(
        (out_height - dilated_kernel + 2 * padding) / stride + 1)
    in_width = math.ceil(
        (out_width - dilated_kernel + 2 * padding) / stride + 1)
    return in_height, in_width

def conv_transpose_out_dim(in_height, in_width, kernel_size,
                          stride=1, padding=0, dilation=1):
    # dilated_kernel = dilation * (kernel_size - 1)
    dilated_kernel = kernel_size
    out_height = math.ceil(
        (in_height - 1) * stride - 2 * padding + dilated_kernel)
    out_width = math.floor(
        (in_width - 1) * stride - 2 * padding + dilated_kernel)
    return out_height, out_width

if opt.activation == 'lrelu':
    activation = F.leaky_relu
elif opt.activation == 'tanh':
    activation = F.tanh
else:
    raise Exception("Activation was not specified properly.")

eps = 1e-2
class Transition(nn.Module):
    def __init__(self, hidden_dim):
        super(Transition, self).__init__()
        self.dim = hidden_dim

        self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim)
                                     for _ in range(4)])
        # self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim)
        #                              for _ in range(2)])

        # self.l1 = nn.Linear(self.dim, self.dim)
        self.lin_mu = nn.Linear(self.dim, self.dim)
        self.lin_sigma = nn.Linear(self.dim, self.dim)

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
            current = F.tanh(current)

        # hidden = F.tanh(self.l1(input))
        mu = self.lin_mu(current)
        # mu = 10 * F.tanh(self.lin_mu(current) / 10)
        sigma = Variable(torch.ones(mu.size()).type_as(mu.data) / 10)
        # sigma = F.softplus(self.lin_sigma(current)) + 1e-2
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

        list(self.lin_mu.parameters())[1].data.normal_(0.5, 0.01)

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
            current = F.tanh(current)

        mu_preactivation = self.lin_mu(current)
        # mu = F.sigmoid(mu_preactivation) + 0.1 * mu_preactivation
        mu = F.leaky_relu(mu_preactivation)

        # sigma = F.sigmoid(self.lin_sigma(current)) + 3e-2
        sigma = Variable(torch.ones(mu.size()).type_as(mu.data) / 10)
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


# [(7, 7), (9, 9), (17, 17), (33, 33), (65, 65), [128, 128]]
class DCGANGenerator(nn.Module):
    def __init__(self, hidden_dim, output_dims):
        super(DCGANGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims

        # self.planes[0] will be set automatically so that it is
        # similar to but larger than hidden_dim
        # self.planes = [None, 64, 64, 128, output_dims[0] * 2]
        # self.kernels = [None, 3, 3, 3, 3]

        # DCGAN numbers
        ngf = 64
        self.planes = [None,
                       ngf * 8,
                       ngf * 4,
                       ngf * 2,
                       ngf,
                       output_dims[0] * 2]
        self.kernels = [None, 4, 4, 4, 4, 4]
        self.pads = [None, 0, 1, 1, 1, 1]
        self.strides = [None, 1, 2, 2, 2, 2]


        # self.planes = [1, 64, output_dims[0] * 2]
        # self.kernels = [None, 3, 3]


        self.in_dims = [output_dims[1:]]
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

        self.convs = nn.ModuleList()

        # print(self.in_dims)
        for l in range(1, len(self.planes)):
            in_height, in_width = self.in_dims[l]
            self.convs.append(nn.ConvTranspose2d(self.planes[l-1],
                                        self.planes[l],
                                        self.kernels[l],
                                        stride=self.strides[l],
                                        padding=self.pads[l]))

        self.convs[-1].bias.data.add_(0.5)

    def forward(self, input):
        current = input
        for lin in self.lins:
            current = lin(current)
            current = F.relu(current)

        current = current.resize(current.size(0),
                                 self.planes[0],
                                 *self.in_dims[0])
        for conv in self.convs:
            current = F.relu(current)
            # current = self.upsample(current)
            current = conv(current)

        mu = F.leaky_relu(current[:, : int(current.size(1) / 2)])
        sigma = Variable(torch.ones(mu.size()).type(dtype) * opt.output_var)
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


class DCGANInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(DCGANInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        ndf = 64
        self.planes = [ndf,
                       ndf * 2,
                       ndf * 4,
                       ndf * 8,
                       ndf // 2]
        self.kernels = [4, 4, 4, 4, 4]
        self.strides = [2, 2, 2, 2, 1]
        self.pads = [1, 1, 1, 1, 0]


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
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        # self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
        #                              for _ in range(2)])


        self.lin_mu = nn.Linear(hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(hidden_dim, self.hidden_dim)


    def forward(self, x_t, z_prev):
        current = x_t
        for conv in self.convs:
            current = conv(current)
            current = F.leaky_relu(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        current = F.leaky_relu(current)

        joined = torch.cat([current, z_prev], 1)
        new_hidden = F.leaky_relu(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

# class RecurrentInferenceCell(nn.Module):
#     def __init__(self, latent_dim, hidden_dim):
#         super(RecurrentInferenceCell, self).__init__()
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim
#
#         self.recurrent = nn.GRUCell(latent_dim, hidden_dim)
#         self.output
#
#
#     def forward(previous, h):
#         self.recurrent(previous, h)
#


class RecurrentInference(nn.Module):
    def __init__(self, input_dims, latent_dim, n_latents):
        super(RecurrentInference, self).__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.n_latents = n_latents

        self.meanfield = ConvInference(input_dims)
        self.recurrent = nn.GRUCell(latent_dim, latent_dim * n_latents)

    def forward(x_t, prior):
        """
        meanfield prediction incorporates information from two things:
        1. the current image x_t
        2. the prior's prediction for each latent z1...zk

        we will use a lightweight recurrent network to adjust the posterior
        of later latents based on the sampled values of the earlier latents
        """
        h = self.meanfield(x_t, prior)[0]

        # we'll use the meanfield prediction as the posterior for z1
        previous_posterior = h[:, : latent_dim]
        posterior = [previous_posterior]
        for z in prior:
            previous_sample = sample(previous_posterior)
            new_posterior, h = self.recurrent(previous_sample, h)
            posterior.append(new_posterior)

            previous_posterior = new_posterior


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

class DCGANFirstInference(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(DCGANFirstInference, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim

        ndf = 64
        self.planes = [ndf,
                       ndf * 2,
                       ndf * 4,
                       ndf * 8,
                       ndf // 2]
        self.kernels = [4, 4, 4, 4, 4]
        self.strides = [2, 2, 2, 2, 1]
        self.pads = [1, 1, 1, 1, 0]

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
                                               padding=self.pads[l],
                                               stride=self.strides[l])))
        self.convs = nn.ModuleList()
        for l in range(len(self.planes)):
            in_planes, in_height, in_width = self.out_dims[l]
            self.convs.append(nn.Conv2d(in_planes,
                                        int(self.planes[l]),
                                        self.kernels[l],
                                        padding=self.pads[l],
                                        stride=self.strides[l]))

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
            current = F.leaky_relu(current)
        current = current.resize(current.size(0), prod(self.out_dims[-1]))
        current = self.input_lin(current)
        new_hidden = F.leaky_relu(current)

        # joined = torch.cat([current, z_prev], 1)
        # new_hidden = F.tanh(self.joint_lin(joined))

        # mu = 10 * F.tanh(self.lin_mu(new_hidden) / 10)
        mu = self.lin_mu(new_hidden)
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)


class GaussianKLD(nn.Module):
    def forward(self, q, p):
        (mu_q, sigma_q) = q
        (mu_p, sigma_p) = p
        mu_q = batch_flatten(mu_q)
        sigma_q = batch_flatten(sigma_q)
        mu_p = batch_flatten(mu_p)
        sigma_p = batch_flatten(sigma_p)


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
        mu = batch_flatten(mu)
        sigma = batch_flatten(sigma)
        target = batch_flatten(target)

        # sigma = Variable(torch.ones(sigma.size()).type_as(sigma.data) / 10)

        a = torch.sum(torch.log(sigma), 1)
        diff = (target - mu)
        b = torch.sum(torch.pow(diff, 2) / sigma, 1)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()
