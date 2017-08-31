import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import scipy.misc
import random
import math
import os
from PIL import Image
import progressbar
import logging
import shutil
import time
import traceback

from util import *

from modules import *
from models import *
from params import *
from setup import *
from generations import *

from covariance import construct_covariance

print("Tensor type: ", dtype)

torch.manual_seed(opt.seed)
random.seed(opt.seed)

# ------- load the model first, as this affects the paths used --------
if opt.load is not None:
    i, model = load_checkpoint(opt)
else:
    i = 0
    if opt.tiny:
        model = IndependentModel(opt.latents,
                                 opt.latent_dim,
                                 opt.image_width,
                                 transition=Transition,
                                 first_inference=TinyDCGANFirstInference,
                                 inference=TinyDCGANInference,
                                 generator=TinyDCGANGenerator).type(dtype)
    else:
        model = IndependentModel(opt.latents,
                                 opt.latent_dim,
                                 opt.image_width).type(dtype)

opt.save = 'results/' + opt.name
print(model)

# -------- take care of logging, cleaning out the folder, etc -------
if opt.git:
    git_hash = make_pointer_commit(
        opt.name,
        message=json.dumps(vars(opt), indent=4, sort_keys=True),
        commit_if_dirty=opt.commit)
    opt.git_hash = git_hash
else:
    opt.git_hash = get_commit_hash()

make_result_folder(opt, opt.save)
write_options(opt, opt.save)

logging.basicConfig(filename = opt.save + "/results.csv",
                    level = logging.DEBUG,
                    format = "%(message)s")
logging.debug(("step,loss,nll,divergence,prior divergence,trans divergence,"
               "qvarmin,qvarmean,qvarmax,"
               "pvarmin,pvarmean,pvarmax,"
               "grad norm,ms/seq,lr"))

# --------- load a dataset ------------------------------------
train_data, test_data, load_workers = load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=load_workers,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=load_workers,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)
print("Number of training sequences (with overlap): " + str(len(train_data)))
print("Number of testing sequences (with overlap): " + str(len(test_data)))
# ------------------------------------

KL = GaussianKLD().type(dtype)
LL = GaussianLL().type(dtype)
mse = nn.MSELoss().type(dtype)
l1 = nn.L1Loss().type(dtype)

optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr)

sequence = None
if opt.sanity:
    opt.max_steps = 10 * opt.batch_size
    opt.print_every = 10 * opt.batch_size

# make opt.print_every a multiple of batch_size
opt.print_every = (opt.print_every // opt.batch_size) * opt.batch_size

# initialize statistics
mean_loss = 0
mean_divergence = 0
mean_nll = 0
mean_prior_div = 0
mean_trans_div = 0
mean_grad_norm = 0
q_var_means = []
q_var_min = 100
q_var_max = -100

p_var_means = []
p_var_min = 100
p_var_max = -100
last_covariance = 0

random_mean_vars = []
norandom_mean_vars = []

progress = progressbar.ProgressBar(max_value=opt.print_every)
while i < opt.max_steps:
    for sequence in train_loader:
        i += opt.batch_size
        # pdb.set_trace()

        sequence = normalize_data(opt, dtype, sequence)

        generations, nll, divergences, batch_z_vars = model(sequence,
                                              motion_weight=opt.motion_weight)
        seq_divergence, seq_prior_div, seq_trans_div = divergences
        mean_divergence += seq_divergence.data[0]
        mean_prior_div += seq_prior_div.data[0]
        mean_trans_div += seq_trans_div.data[0]
        mean_nll += nll.data[0]

        p_vars, q_vars = batch_z_vars

        q_var_means.append(sum([v.data.mean() for v in q_vars]) / \
                               len(q_vars))
        q_var_min = min(*[v.data.min() for v in q_vars], q_var_min)
        q_var_max = max(*[v.data.max() for v in q_vars], q_var_max)

        if opt.seq_len > 1:
            p_var_means.append(sum([v.data.mean() for v in p_vars]) / \
                               len(p_vars))
            p_var_min = min(*[v.data.min() for v in p_vars], p_var_min)
            p_var_max = max(*[v.data.max() for v in p_vars], p_var_max)

        # pdb.set_trace()

        # accumulate the variances for randomizing and nonrandomizing frames
        if opt.data == 'random_balls':
            for t in range(opt.seq_len - 1):
                for b in range(opt.batch_size):
                    mean = p_vars[t][b].data.mean()
                    if randomize[t][b] == 1:
                        random_mean_vars.append(mean)
                    else:
                        norandom_mean_vars.append(mean)


        kl_penalty = seq_divergence * opt.kl_weight
        if opt.kl_anneal:
            kl_weight = max(0, min(i / opt.kl_anneal_end, 1))
            kl_penalty = kl_weight * kl_penalty

        loss = nll + kl_penalty
        mean_loss += loss.data[0]

        model.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), )

        mean_grad_norm += grad_norm(model)

        if opt.sgld:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data += p.grad.data.clone().normal_(0, opt.lr)
        optimizer.step()

        progress.update(i % opt.print_every)
        is_update_time = (i >= opt.max_steps or
                          (i % opt.print_every == 0 and i > 0))
        if is_update_time:
            progress.finish()
            clear_progressbar()

            if opt.data == 'random_balls':
                print("Variance on randomizing frames: {}".format(
                        sum(random_mean_vars) / len(random_mean_vars)))
                print("Variance on non-randomizing frames: {}".format(
                        sum(norandom_mean_vars) / len(norandom_mean_vars)))
                random_mean_vars = []
                norandom_mean_vars = []



            elapsed_time = progress.end_time - progress.start_time
            elapsed_seconds = elapsed_time.total_seconds()

            batches = opt.print_every / opt.batch_size
            if opt.seq_len > 1:
                p_var_mean = sum(p_var_means) / len(p_var_means)
            else:
                p_var_mean = 0
            q_var_mean = sum(q_var_means) / len(q_var_means)
            log_values = (i,
                          mean_loss / batches,
                          mean_nll / batches,
                          mean_divergence / batches,
                          mean_prior_div / batches,
                          mean_trans_div / batches,
                          q_var_min, 
                          q_var_mean,
                          q_var_max,
                          p_var_min, 
                          p_var_mean,
                          p_var_max,
                          mean_grad_norm / batches,
                          elapsed_seconds / opt.print_every * 1000,
                          opt.lr)
            # log_values = (i,
            #               mean_loss / batches,
            #               mean_nll / batches,
            #               mean_divergence / batches,
            #               mean_prior_div / batches,
            #               mean_trans_div / batches,
            #               mean_grad_norm / batches,
            #               elapsed_seconds / opt.print_every * 1000,
            #               opt.lr)
            print(("Step: {:8d}, Loss: {:8.3f}, NLL: {:8.3f}, "
                   "Divergence: {:8.3f}, "
                   "Prior divergence: {:8.3f}, "
                   "Trans divergence: {:8.3f}, "
                   "q(z) vars: [{:7.3f}, {:7.3f}, {:7.3f}], "
                   "p(z) vars: [{:7.3f}, {:7.3f}, {:7.3f}], "
                   "Grad norm: {:10.3f}, "
                   "ms/seq: {:6.2f}").format(*log_values[:-1]))

            # make list of n copies of format string, then format
            format_string = ",".join(["{:.8e}"]*len(log_values))
            logging.debug(format_string.format(*log_values))

            mean_loss = 0
            mean_divergence = 0
            mean_prior_div = 0
            mean_trans_div = 0
            mean_grad_norm = 0
            mean_nll = 0
            q_var_means = []
            q_var_min = 100
            q_var_max = -100
            p_var_means = []
            p_var_min = 100
            p_var_max = -100
            try:
                save_all_generations(i, model, sequence, generations)
            except:
                traceback.print_exc()

        if i >= opt.max_steps or (i % opt.save_every == 0 and i > 0):
            save_dict = {
                    'model': model,
                    'opt': vars(opt),
                    'i': i,
                }
            torch.save(save_dict, opt.save + '/model.t7')

        if opt.seq_len > 1:
            if i == opt.max_steps or i - last_covariance > opt.cov_every:
                construct_covariance(opt.save + '/covariance/', 
                                     model, train_loader, 2000,
                                     label="train_" + str(i))
                construct_covariance(opt.save + '/covariance/',
                                     model, test_loader, 2000,
                                     label="test_" + str(i))
                last_covariance = i

        # learning rate decay
        # 0.985 every 320K -> ~0.2 at 1,000,000 steps
        if not opt.no_lr_decay and i % 320000 == 0 and i > 0:
            opt.lr = opt.lr * 0.985
            print("Decaying learning rate to: ", opt.lr)
            set_lr(optimizer, opt.lr)

        if is_update_time:
            progress = progressbar.ProgressBar(max_value=opt.print_every)

        if i >= opt.max_steps:
            break
