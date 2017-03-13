import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

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
        mu = F.tanh(self.lin_mu(hidden))
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
                                     for _ in range(0)])
        self.lin_mu = nn.Linear(self.hidden_dim, self.output_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer(current)
            current = F.relu(current)

        mu = F.sigmoid(self.lin_mu(current))
        sigma = F.sigmoid(self.lin_sigma(current)) + eps
        return (mu, sigma)

class Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Inference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.joint_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(0)])

        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t, z_prev):
        embedded = F.tanh(self.input_lin(x_t))
        joined = torch.cat([embedded, z_prev], 1)
        new_hidden = F.tanh(self.joint_lin(joined))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)

        mu = F.tanh(self.lin_mu(new_hidden))
        sigma = F.sigmoid(self.lin_sigma(new_hidden)) + eps
        return (mu, sigma)

class FirstInference(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FirstInference, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                     for _ in range(0)])

        self.lin_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)


    def forward(self, x_t):
        new_hidden = F.tanh(self.input_lin(x_t))
        for layer in self.layers:
            new_hidden = layer(new_hidden)
            new_hidden = F.tanh(new_hidden)

        mu = F.tanh(self.lin_mu(new_hidden))
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

        a = torch.sum(torch.log(sigma), 1)
        diff = (target - mu)
        b = torch.sum(torch.pow(diff, 2) / sigma, 1)
        c = mu.size(1) * math.log(2*math.pi)
        log_likelihoods = -0.5 * (a + b + c)
        return log_likelihoods.mean()
