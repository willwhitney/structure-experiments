import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import scipy.misc
import random
import math
import os
import socket
from PIL import Image

from modules import *
from env import *

print("Running on machine: ", socket.gethostname())
if socket.gethostname() == 'zaan':
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

name = os.environ['CUDA_VISIBLE_DEVICES']
if not os.path.exists(name):
    os.mkdir(name)

def sample(p):
    (mu, sigma) = p
    noise = torch.normal(torch.zeros(mu.size()), torch.ones(sigma.size())).type(dtype)
    noise = Variable(noise)
    return mu + sigma * noise



gen = DataGenerator()
batch_size = 1000
data_dim = gen.start().render().nelement()
hidden_dim = 200
T = 10

KL = GaussianKLD().type(dtype)
# KL1 = GaussianKLD1().type(dtype)
LL = GaussianLL().type(dtype)
mse = nn.MSELoss().type(dtype)
l1 = nn.L1Loss().type(dtype)


# z1_prior = (Variable(torch.zeros(batch_size, hidden_dim).type(dtype)),
#             Variable(torch.ones(batch_size, hidden_dim).type(dtype)))


class WholeModel(nn.Module):
    def __init__(self):
        super(WholeModel, self).__init__()
        self.z1_prior = (
            Variable(torch.zeros(batch_size, hidden_dim).type(dtype)),
            Variable(torch.ones(batch_size, hidden_dim).type(dtype))
        )

        self.transition = Transition(hidden_dim)
        self.first_inference = FirstInference(data_dim, hidden_dim)
        self.inference = Inference(data_dim, hidden_dim)
        self.generator = Generator(hidden_dim, data_dim)

    def forward(self, sequence):
        loss = Variable(torch.zeros(1).type(dtype))
        generations = []

        inferred_z_post = self.first_inference(sequence[0])
        z_prior = self.z1_prior
        for t in range(len(sequence)):
            divergence = KL(inferred_z_post, z_prior)
            loss = loss + divergence

            z_sample = sample(inferred_z_post)
            # z_sample = inferred_z_post[0]

            gen_dist = self.generator(z_sample)
            # log_likelihood = - mse(gen_dist[0], sequence[t])
            log_likelihood = LL(gen_dist, sequence[t])
            loss = loss - log_likelihood

            generations.append(gen_dist)

            if t < len(sequence) - 1:
                z_prior = self.transition(z_sample)
                inferred_z_post = self.inference(sequence[t+1], z_sample)
                # inferred_z_post = self.first_inference(sequence[t+1])

        return generations, loss / len(sequence)

class IndependentModel(nn.Module):
    def __init__(self):
        super(WholeModel, self).__init__()
        self.z1_prior = (
            Variable(torch.zeros(batch_size, hidden_dim).type(dtype)),
            Variable(torch.ones(batch_size, hidden_dim).type(dtype))
        )

        self.transition = Transition(hidden_dim)
        self.first_inference = FirstInference(data_dim, hidden_dim)
        self.inference = Inference(data_dim, hidden_dim)
        self.generator = Generator(hidden_dim, data_dim)

    def forward(self, sequence):
        loss = Variable(torch.zeros(1).type(dtype))
        generations = []

        inferred_z_post = self.first_inference(sequence[0])
        z_prior = self.z1_prior
        for t in range(len(sequence)):
            divergence = KL(inferred_z_post, z_prior)
            loss = loss + divergence

            z_sample = sample(inferred_z_post)
            # z_sample = inferred_z_post[0]

            gen_dist = self.generator(z_sample)
            # log_likelihood = - mse(gen_dist[0], sequence[t])
            log_likelihood = LL(gen_dist, sequence[t])
            loss = loss - log_likelihood

            generations.append(gen_dist)

            if t < len(sequence) - 1:
                z_prior = self.transition(z_sample)
                inferred_z_post = self.inference(sequence[t+1], z_sample)
                # inferred_z_post = self.first_inference(sequence[t+1])

        return generations, loss / len(sequence)

def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return
    for p in parameters:
        p.grad.data.mul_(clip_coef)

def make_real_seq(length):
    sequence = [torch.zeros(batch_size, gen.size**2 * 3) for _ in range(length)]
    for batch in range(batch_size):
        sequence[0][batch] = gen.start().render().view(gen.size**2 * 3)
        for i in range(1, length):
            sequence[i][batch] = gen.step().render().view(gen.size**2 * 3)

    return [Variable(x.type(dtype)) for x in sequence]

def make_seq(length, dim):
    sequence = [torch.zeros(batch_size, dim).normal_(0.5, 0.1)]
    for i in range(1, length):
        noise = torch.zeros(batch_size, dim) # torch.normal([0.0, 0.0], [0.1, 0.1])
        sequence.append(sequence[i-1] + noise)
    sequence = [Variable(x.type(dtype)) for x in sequence]
    return sequence

def generate(model, steps):
    priming_steps = 2
    generations = []

    priming = make_real_seq(priming_steps)
    latent = sample(model.first_inference(priming[0]))
    generations.append(model.generator(latent))
    for t in range(1, priming_steps):
        latent = sample(model.inference(priming[t], latent))
        generations.append(model.generator(latent))

    for t in range(steps - priming_steps):
        latent = sample(model.transition(latent))
        generations.append(model.generator(latent))
    # print(generations)
    return generations


model = WholeModel().type(dtype)
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3)

print(model)

mean_loss = 0
n_steps = int(1e5)
for i in range(n_steps):
    # sequence = make_seq(1, data_dim)
    # sequence = make_real_seq(np.random.geometric(0.1))
    sequence = make_real_seq(1)

    generations, loss = model(sequence)
    mean_loss += loss.data[0]

    model.zero_grad()
    loss.backward()
    # print(list(model.first_inference.parameters())[0].norm())

    # clip_grad_norm(model.parameters(), 10)
    optimizer.step()

    k = 100
    if i % k == 0:
        print("Step: ", i, "\tLoss: ", mean_loss / k)
        # print(list(model.transition.parameters())[0].norm())
        mean_loss = 0

    if i % 1000 == 0 or i == n_steps - 1:
        gen_data = [(gen[0].data, gen[1].data) for gen in generations]

        seq_data = [x.data for x in sequence]

        # %matplotlib inline
        # show(seq_data[0].view(4,4,3))
        for j in range(5):
            timesteps = len(seq_data)
            result = torch.zeros(2 * gen.size, timesteps * gen.size, 3)
            for t in range(timesteps):
                mu, sigma = gen_data[t]
                mu = mu[j]
                result[:gen.size, gen.size*t:gen.size*(t+1)] = seq_data[t][j].view(gen.size,gen.size,3)
                result[gen.size:, gen.size*t:gen.size*(t+1)] = mu.view(gen.size,gen.size,3)
            scipy.misc.imsave(
                name + '/result_'+str(j)+'.png',
                result.numpy())

        samples = generate(model, 20)
        print("Samples: ", len(samples))
        mus = [x[0].data for x in samples]
        for j in range(5):
            result = torch.zeros(gen.size, len(samples) * gen.size, 3)
            for t in range(len(samples)):
                mu = mus[t][j]
                result[:, gen.size*t:gen.size*(t+1)] = mu.view(gen.size,gen.size,3)
            scipy.misc.imsave(
                name + '/gen_'+str(j)+'.png',
                result.numpy())

        prior_sample = sample(model.z1_prior)
        image_dist = model.generator(prior_sample)
        image_sample = image_dist[0].resize(32 * gen.size, gen.size, 3)
        scipy.misc.imsave(name + '/prior_samples.png', image_sample.data.cpu().numpy())
