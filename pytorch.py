import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
from params import opt, dtype
from setup import *
from generations import *
from bookkeeper import Bookkeeper

from covariance import construct_covariance

print("Tensor type: ", dtype)

torch.manual_seed(opt.seed)
random.seed(opt.seed)

# ------- load the checkpoint first, as this affects the paths used --------
if opt.load is not None:
    i, model = load_checkpoint(opt)
else:
    if opt.model == 'independent':
        modeltype = IndependentModel
    elif opt.model == 'deterministic':
        modeltype = DeterministicModel

    i = 0
    if opt.tiny:
        model = modeltype(opt.latents,
                                 opt.latent_dim,
                                 opt.image_width,
                                 transition=Transition,
                                 first_inference=TinyDCGANFirstInference,
                                 inference=TinyDCGANInference,
                                 generator=TinyDCGANGenerator).type(dtype)
    else:
        model = modeltype(opt.latents,
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
               "ms/seq,lr"))

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


# --------- configure optimizer and LR decay ---------------
optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr)

lr_lambda = lambda epoch: opt.lr_decay ** (i / 320000)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------- set up bookkeeper and its functions ---------------
sequence, generations = None, None

def reset_state(state):
    state['mean_loss'] = 0
    state['mean_divergence'] = 0
    state['mean_nll'] = 0
    state['mean_prior_div'] = 0
    state['mean_trans_div'] = 0
    state['mean_grad_norm'] = 0

    state['q_var_means'] = []
    state['q_var_min'] = 100
    state['q_var_max'] = -100

    state['p_var_means'] = []
    state['p_var_min'] = 100
    state['p_var_max'] = -100
    state['progress'] = progressbar.ProgressBar(max_value=opt.print_every)
    return state

def update_reducer(step, state, updates):
    state['progress'].update(step % opt.print_every)

    # seq_divergence, seq_prior_div, seq_trans_div = updates['divergences']
    state['mean_divergence'] += updates['seq_divergence'].data[0]
    state['mean_prior_div'] += updates['seq_prior_div'].data[0]
    state['mean_trans_div'] += updates['seq_trans_div'].data[0]
    state['mean_nll'] += updates['seq_nll'].data[0]

    loss = updates['seq_nll'].data[0] + updates['seq_divergence'].data[0]
    state['mean_loss'] += loss

    p_vars = updates['prior_variances']
    q_vars = updates['posterior_variances']

    state['q_var_means'].append(mean_of_means(q_vars))
    state['q_var_min'] = min(*[v.data.min() for v in q_vars],
                             state['q_var_min'])
    state['q_var_max'] = max(*[v.data.max() for v in q_vars],
                             state['q_var_max'])

    if opt.seq_len > 1:
        state['p_var_means'].append(mean_of_means(p_vars))
        state['p_var_min'] = min(*[v.data.min() for v in p_vars],
                                 state['p_var_min'])
        state['p_var_max'] = max(*[v.data.max() for v in p_vars],
                                 state['p_var_max'])

def make_log(step, state):
    state['progress'].finish()
    clear_progressbar()

    elapsed_time = state['progress'].end_time - state['progress'].start_time
    elapsed_seconds = elapsed_time.total_seconds()
    batches = opt.print_every / opt.batch_size

    q_var_mean = sum(state['q_var_means']) / len(state['q_var_means'])
    if opt.seq_len > 1:
        p_var_mean = sum(state['p_var_means']) / len(state['p_var_means'])
    else:
        p_var_mean = 0

    log_values = (step,
                  state['mean_loss'] / batches,
                  state['mean_nll'] / batches,
                  state['mean_divergence'] / batches,
                  state['mean_prior_div'] / batches,
                  state['mean_trans_div'] / batches,
                  state['q_var_min'],
                  q_var_mean,
                  state['q_var_max'],
                  state['p_var_min'],
                  p_var_mean,
                  state['p_var_max'],
                  # mean_grad_norm / batches,
                  elapsed_seconds / opt.print_every * 1000,
                  scheduler.get_lr()[0])

    print(("Step: {:8d}, Loss: {:8.3f}, NLL: {:8.3f}, "
           "Divergence: {:8.3f}, "
           "Prior divergence: {:8.3f}, "
           "Trans divergence: {:8.3f}, "
           "q(z) vars: [{:7.3f}, {:7.3f}, {:7.3f}], "
           "p(z) vars: [{:7.3f}, {:7.3f}, {:7.3f}], "
           # "Grad norm: {:10.3f}, "
           "ms/seq: {:6.2f}").format(*log_values[:-1]))

    # make list of n copies of format string, then format
    format_string = ",".join(["{:.8e}"]*len(log_values))
    logging.debug(format_string.format(*log_values))
    reset_state(state)

    try:
        save_all_generations(step, model, sequence, generations)
    except:
        traceback.print_exc()

def save_checkpoint(step, state):
    save_dict = {
        'model': model,
        'opt': vars(opt),
        'i': step,
    }
    torch.save(save_dict, opt.save + '/model.t7')

def make_covariance(step, state):
    if opt.seq_len > 1:
        construct_covariance(opt.save + '/covariance/',
                             model, train_loader, 10,
                             label="train_" + str(step))
        construct_covariance(opt.save + '/covariance/',
                             model, test_loader, 10,
                             label="test_" + str(step))

bookkeeper = Bookkeeper(i, reset_state({}), update_reducer)
bookkeeper.every(opt.print_every, make_log)
bookkeeper.every(opt.save_every, save_checkpoint)
bookkeeper.every(opt.cov_every, make_covariance)

random_mean_vars = []
norandom_mean_vars = []

while i < opt.max_steps:
    for sequence in train_loader:
        i += opt.batch_size

        sequence = normalize_data(opt, dtype, sequence)
        output = model(sequence, motion_weight=opt.motion_weight)
        generations = output['generations']
        bookkeeper.update(i, output)

        nll = output['seq_nll']
        # seq_divergence = output['seq_divergence']
        seq_divergence = output['seq_trans_div']
        # seq_divergence = Variable(torch.zeros(1)).type(dtype)

        # accumulate the variances for randomizing and nonrandomizing frames
        if opt.data == 'random_balls':
            for t in range(opt.seq_len - 1):
                for b in range(opt.batch_size):
                    mean = p_vars[t][b].data.mean()
                    if randomize[t][b] == 1:
                        random_mean_vars.append(mean)
                    else:
                        norandom_mean_vars.append(mean)

        # scale the KL however appropriate
        kl_penalty = seq_divergence * opt.kl_weight
        if opt.kl_anneal:
            kl_weight = max(0, min(i / opt.kl_anneal_end, 1))
            kl_penalty = kl_weight * kl_penalty

        loss = nll + kl_penalty

        model.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        if opt.data == 'random_balls' and i % opt.print_every == 0 and i > 0:
            print("Variance on randomizing frames: {}".format(
                    sum(random_mean_vars) / len(random_mean_vars)))
            print("Variance on non-randomizing frames: {}".format(
                    sum(norandom_mean_vars) / len(norandom_mean_vars)))
            random_mean_vars = []
            norandom_mean_vars = []

        if i >= opt.max_steps:
            break
