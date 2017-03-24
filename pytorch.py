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
from PIL import Image
import argparse

from modules import *
from models import *
from env import *
from util import *
from params import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=os.environ['CUDA_VISIBLE_DEVICES'])
parser.add_argument('--lr', default=3e-4, type=float)
opt = parser.parse_args()
opt.save = 'results/' + opt.name

print("Tensor type: ", dtype)

# name = os.environ['CUDA_VISIBLE_DEVICES'] if name not in locals()
if not os.path.exists(opt.save):
    os.makedirs(opt.save)


gen = DataGenerator()
data_dim = gen.start().render().nelement()
T = 10

KL = GaussianKLD().type(dtype)
LL = GaussianLL().type(dtype)
mse = nn.MSELoss().type(dtype)
l1 = nn.L1Loss().type(dtype)

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

# model = WholeModel(gen.size).type(dtype)
model = WholeModel(gen.size).type(dtype)
optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr)

print(model)

mean_loss = 0
mean_divergence = 0
z_vars = []
n_steps = int(1e6)
for i in range(n_steps):
    # sequence = make_seq(1, data_dim)
    sequence = make_real_seq(5)

    # sequence = make_real_seq(np.random.geometric(0.3))
    # while len(sequence) < 3:
    #     sequence = make_real_seq(np.random.geometric(0.3))

    generations, loss, divergence, z_var = model(sequence)
    mean_loss += loss.data[0]
    mean_divergence += divergence
    if z_var > 0:
        z_vars.append(z_var)

    model.zero_grad()
    loss.backward()

    # torch.nn.utils.clip_grad_norm(model.parameters(), 10)
    optimizer.step()

    k = 1000
    if i % k == 0 and i > 0:
        if len(z_vars) == 0:
            z_vars = [999]
        print(
            ("Step: {:8d}, Loss: {:8.3f}, NLL: {:8.3f}, "
             "Divergence: {:8.3f}, z variance: {:8.3f}").format(
                i,
                mean_loss / k,
                (mean_loss - mean_divergence) / k,
                mean_divergence / k,
                sum(z_vars) / len(z_vars))
        )
        # print("Step: ", i,
        #       "\tLoss: ", mean_loss / k,
        #       "\tNLL: ", (mean_loss - mean_divergence) / k,
        #       "\tDivergence: ", mean_divergence / k,
        #       "\tz variance: ", sum(z_vars) / len(z_vars))
        mean_loss = 0
        mean_divergence = 0
        z_vars = []

    if i % 1000 == 0 or i == n_steps - 1:

        # show some results from the latest batch
        mus_data = [gen[0].data for gen in generations]
        seq_data = [x.data for x in sequence]
        for j in range(5):
            mus = [x[j].view(3,gen.size,gen.size) for x in mus_data]
            truth = [x[j].view(3,gen.size,gen.size) for x in seq_data]
            save_tensors_image(opt.save + '/result_'+str(j)+'.png', [truth, mus])

        # show sequence generations
        samples = model.generate(make_real_seq(2), 20, True)
        mus = [x.data for x in samples]
        for j in range(5):
            mu = [x[j].view(3,gen.size,gen.size) for x in mus]
            save_tensors_image(opt.save + '/gen_'+str(j)+'.png', mu)

        # show max-likelihood generations
        samples = model.generate(make_real_seq(2), 20, False)
        mus = [x.data for x in samples]
        for j in range(5):
            mu = [x[j].view(3,gen.size,gen.size) for x in mus]
            save_tensors_image(opt.save + '/ml_'+str(j)+'.png', mu)

        # show samples from the first-frame prior
        if not isinstance(model, MSEModel):
            prior_sample = sample(model.z1_prior)
            image_dist = model.generator(prior_sample)
            image_sample = image_dist[0].resize(32, 3, gen.size, gen.size)
            image_sample = [[image] for image in image_sample]
            save_tensors_image(opt.save + '/prior_samples.png', image_sample)
