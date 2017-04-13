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
import progressbar
import logging

from util import *

from modules import *
from models import *
from env import *
from params import *


print("Tensor type: ", dtype)

logging.basicConfig(filename = opt.save + "/results.csv",
                    level = logging.DEBUG,
                    format = "%(message)s")
logging.debug(("step,loss,nll,divergence,prior divergence,"
               "trans divergence,grad norm,ms/seq"))

gen = DataGenerator()
data_dim = gen.start().render().nelement()

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

# model = VAEModel(100, gen.size).type(dtype)
# model = IndependentModel(3, 10, gen.size).type(dtype)
model = IndependentModel(opt.latents, opt.latent_dim, gen.size).type(dtype)
optimizer = optim.Adam(
# optimizer = optim.RMSprop(
    model.parameters(),
    lr=opt.lr)

print(model)

mean_loss = 0
mean_divergence = 0
mean_nll = 0
mean_prior_div = 0
mean_trans_div = 0
mean_grad_norm = 0
z_var_means = []
z_var_min = 1e6
z_var_max = -1
n_steps = int(1e7)

k = 1000
progress = progressbar.ProgressBar(max_value=k)
for i in range(n_steps):
    sequence = make_real_seq(5)

    # sequence = make_real_seq(np.random.geometric(0.3))
    # while len(sequence) < 3:
    #     sequence = make_real_seq(np.random.geometric(0.3))

    generations, nll, divergences = model(sequence)
    (seq_divergence, seq_prior_div, seq_trans_div) = divergences
    mean_divergence += seq_divergence.data[0]
    mean_prior_div += seq_prior_div.data[0]
    mean_trans_div += seq_trans_div.data[0]
    mean_nll += nll.data[0]

    # var_min, var_mean, var_max = batch_z_vars
    # z_var_means.append(var_mean)
    # z_var_min = min(var_min, z_var_min)
    # z_var_max = max(var_max, z_var_max)

    if not opt.no_kl_annealing:
        kl_penalty = max(0, min(i / 100000, 1)) * seq_divergence
    else:
        kl_penalty = seq_divergence

    loss = nll + kl_penalty
    mean_loss += loss.data[0]

    model.zero_grad()
    loss.backward()

    mean_grad_norm += grad_norm(model)

    # torch.nn.utils.clip_grad_norm(model.parameters(), 10)
    if not opt.no_sgld:
        for p in model.parameters():
            # import ipdb; ipdb.set_trace()
            if p.grad is not None:
                p.grad.data += p.grad.data.clone().normal_(0, opt.lr)
    optimizer.step()

    progress.update(i%k)
    if i % k == 0 and i > 0:
        progress.finish()
        clear_progressbar()

        elapsed_time = progress.end_time - progress.start_time
        elapsed_seconds = elapsed_time.total_seconds()

        log_values = (i,
                      mean_loss / k,
                      mean_nll / k,
                      mean_divergence / k,
                      mean_prior_div / k,
                      mean_trans_div / k,
                      mean_grad_norm / k,
                      elapsed_seconds / k / batch_size * 1000)
        print(("Step: {:8d}, Loss: {:8.3f}, NLL: {:8.3f}, "
               "Divergence: {:6.3f}, "
               "Prior divergence: {:6.3f}, "
               "Trans divergence: {:6.3f}, "
               "Grad norm: {:7.3f}, "
               "ms/seq: {:6.2f}").format(*log_values))

        # make list of n copies of format string, stitch them together, format
        logging.debug(",".join(["{:.8e}"]*len(log_values)).format(*log_values))

        torch.save(model, opt.save + '/model.t7')
        mean_loss = 0
        mean_divergence = 0
        mean_prior_div = 0
        mean_trans_div = 0
        mean_grad_norm = 0
        mean_nll = 0
        progress = progressbar.ProgressBar(max_value=k)

    # learning rate decay
    # 0.985 every 10K -> ~0.2 at 1,000,000 steps
    if not opt.no_lr_decay and i % 10000 == 0 and i > 0:
        opt.lr = opt.lr * 0.985
        print("Decaying learning rate to: ", opt.lr)
        set_lr(optimizer, opt.lr)
        # optimizer = optim.Adam(
        #     model.parameters(),
        #     lr=opt.lr)

    if i % k == 0 or i == n_steps - 1:

        # save some results from the latest batch
        mus_data = [gen[0].data for gen in generations]
        seq_data = [x.data for x in sequence]
        for j in range(5):
            mus = [x[j].view(3,gen.size,gen.size) for x in mus_data]
            truth = [x[j].view(3,gen.size,gen.size) for x in seq_data]
            save_tensors_image(opt.save + '/result_'+str(j)+'.png', [truth, mus])

        priming = make_real_seq(2)

        # save sequence generations
        samples = model.generate(priming, 20, True)
        mus = [x.data for x in samples]
        for j in range(5):
            mu = [x[j].view(3,gen.size,gen.size) for x in mus]
            save_tensors_image(opt.save + '/gen_'+str(j)+'.png', mu)

        # save max-likelihood generations
        samples = model.generate(priming, 20, False)
        mus = [x.data for x in samples]
        for j in range(5):
            mu = [x[j].view(3,gen.size,gen.size) for x in mus]
            save_tensors_image(opt.save + '/ml_'+str(j)+'.png', mu)

        # save samples from the first-frame prior
        if not isinstance(model, MSEModel):
            prior_sample = sample(model.z1_prior)
            image_dist = model.generator(prior_sample)
            image_sample = image_dist[0].resize(32, 3, gen.size, gen.size)
            image_sample = [[image] for image in image_sample]
            save_tensors_image(opt.save + '/prior_samples.png', image_sample)

        # save ML generations with only one latent evolving
        if isinstance(model, IndependentModel):
            samples = model.generate_independent(priming, 20, False)
            samples = [[x.data for x in sample_row]
                       for sample_row in samples]
            for j in range(5):
                image = [[x[j].view(3,gen.size,gen.size) for x in sample_row]
                          for sample_row in samples]
                save_tensors_image(opt.save + '/ind_ml_'+str(j)+'.png', image)

        # save samples with only one latent evolving
        if isinstance(model, IndependentModel):
            samples = model.generate_independent(priming, 20, True)
            samples = [[x.data for x in sample_row]
                       for sample_row in samples]
            for j in range(5):
                image = [[x[j].view(3,gen.size,gen.size) for x in sample_row]
                          for sample_row in samples]
                save_tensors_image(opt.save + '/ind_gen_'+str(j)+'.png', image)

        # save samples with only one latent randomly sampling
        if isinstance(model, IndependentModel):
            samples = model.generate_variations(priming, 20)
            samples = [[x.data for x in sample_row]
                       for sample_row in samples]
            for j in range(5):
                image = [[x[j].view(3,gen.size,gen.size) for x in sample_row]
                          for sample_row in samples]
                save_tensors_image(opt.save + '/ind_resample_'+str(j)+'.png', image)

        # save samples interpolating between noise near the current latent
        if isinstance(model, IndependentModel):
            samples = model.generate_interpolations(priming, 20)
            samples = [[x.data for x in sample_row]
                       for sample_row in samples]
            for j in range(10):
                image = [[x[j].view(3,gen.size,gen.size) for x in sample_row]
                          for sample_row in samples]
                save_tensors_image(opt.save + '/interp_'+str(j)+'.png', image)
