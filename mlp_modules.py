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