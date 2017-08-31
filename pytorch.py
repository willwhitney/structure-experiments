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
from bookkeeper import Bookkeeper

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
# ------------------------------------


optimizer = optim.Adam(
    model.parameters(),
    lr=opt.lr)

# make opt.print_every a multiple of batch_size
opt.print_every = (opt.print_every // opt.batch_size) * opt.batch_size
bookkeeper = Bookkeeper(i)

random_mean_vars = []
norandom_mean_vars = []

while i < opt.max_steps:
    for sequence in train_loader:
        i += opt.batch_size

        sequence = normalize_data(opt, dtype, sequence)
        # pdb.set_trace()

        outputs = model(sequence, motion_weight=opt.motion_weight)
        bookkeeper.update(i, model, sequence, *outputs)

        _, nll, divergences, *_ = outputs
        seq_divergence, seq_prior_div, seq_trans_div = divergences

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

        if opt.sgld:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data += p.grad.data.clone().normal_(0, opt.lr)
        optimizer.step()

        if opt.data == 'random_balls' and i % opt.print_every == 0 and i > 0:
            print("Variance on randomizing frames: {}".format(
                    sum(random_mean_vars) / len(random_mean_vars)))
            print("Variance on non-randomizing frames: {}".format(
                    sum(norandom_mean_vars) / len(norandom_mean_vars)))
            random_mean_vars = []
            norandom_mean_vars = []

        # learning rate decay
        # 0.985 every 320K -> ~0.2 at 1,000,000 steps
        if not opt.no_lr_decay and i % 320000 == 0 and i > 0:
            opt.lr = opt.lr * 0.985
            print("Decaying learning rate to: ", opt.lr)
            set_lr(optimizer, opt.lr)

        if i >= opt.max_steps:
            break
