import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import pdb

from util import *
from params import *

from dcgan_modules import *
from tinydcgan_modules import *
from mlp_modules import *
from conv_modules import *
from loss_modules import *

# from adversarial_modules import *

if opt.activation == 'lrelu':
    activation = F.leaky_relu
elif opt.activation == 'tanh':
    activation = F.tanh
elif opt.activation == 'selu':
    activation = F.selu
else:
    raise Exception("Activation was not specified properly.")

eps = 1e-5
# class Transition(nn.Module):
#     def __init__(self, hidden_dim, layers=4):
#         super(Transition, self).__init__()
#         self.dim = hidden_dim

#         self.input_lin = nn.Linear(self.dim * 2, self.dim)

#         self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim)
#                                      for _ in range(layers)])

#         self.lin_mu = nn.Linear(self.dim, self.dim)
#         self.lin_sigma = nn.Linear(self.dim, self.dim)

#     def forward(self, inputs):
#         current = self.input_lin(torch.cat(inputs, 1))
#         current = activation(current)
#         for layer in self.layers:
#             current = layer(current)
#             current = activation(current)

#         mu = self.lin_mu(current)
#         sigma = self.lin_sigma(current)
#         return (mu, sigma)


class Transition(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=4):
        super().__init__()
        self.dim = hidden_dim
        self.input_lin = nn.Linear(input_dim * 2, self.dim)

        self.layers = nn.ModuleList([nn.Linear(self.dim, self.dim)
                                     for _ in range(layers)])

        self.lin_mu = nn.Linear(self.dim, input_dim)
        self.lin_sigma = nn.Linear(self.dim, input_dim)

    def forward(self, inputs):
        current = self.input_lin(torch.cat(inputs, 1))
        current = activation(current)
        for layer in self.layers:
            current = layer(current)
            current = activation(current)

        mu = self.lin_mu(current)
        sigma = self.lin_sigma(current)
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
