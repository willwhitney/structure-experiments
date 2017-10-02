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
import ipdb

from util import *

from modules import *
from models import *
from params import *
from setup import *
import generations
from bookkeeper import Bookkeeper

from covariance import construct_covariance

print("Tensor type: ", dtype)

torch.manual_seed(opt.seed)
random.seed(opt.seed)

# ------- load the checkpoint first, as this affects the paths used --------
if opt.load is not None:
    i, model = load_checkpoint(opt, dtype)
else:
    i = 0
    model = PredictionModel(opt.latents,
                            opt.latent_dim, 
                            opt.context_dim, 
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

logging.basicConfig(filename=opt.save + "/results.csv",
                    level=logging.DEBUG,
                    format="%(message)s")
logging.debug(("step,loss,"
               "ms/seq"))

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

# --------- set up bookkeeper and its functions ---------------
sequence, generated_sequence = None, None

def reset_state(state):
    state['loss'] = 0
    state['progress'] = progressbar.ProgressBar(max_value=opt.print_every)
    return state


def update_reducer(step, state, updates):
    state['progress'].update(step % opt.print_every)
    state['loss'] += updates['loss'].data[0]


def make_log(step, state):
    state['progress'].finish()
    clear_progressbar()

    elapsed_time = state['progress'].end_time - state['progress'].start_time
    elapsed_seconds = elapsed_time.total_seconds()
    batches = opt.print_every / opt.batch_size
    log_values = (step,
                  state['loss'] / batches,
                  elapsed_seconds / opt.print_every * 1000)

    print(("Step: {:8d}, Loss: {:8.3f}, "
           "ms/seq: {:6.2f}").format(*log_values))

    # make list of n copies of format string, then format
    format_string = ",".join(["{:.8e}"] * len(log_values))
    logging.debug(format_string.format(*log_values))
    reset_state(state)

    model.eval()
    volatile_sequence = [x.detach() for x in sequence]
    for x in volatile_sequence:
        x.volatile = True
    generations.save_paired_sequence(
        os.path.join(opt.save, "reconstruction", str(step) + '-'),
        volatile_sequence, [volatile_sequence[0]] + generated_sequence)
    generations.save_independent_gen(
        os.path.join(opt.save, "ind_gen", str(step) + '-'),
        model, volatile_sequence, sampling=False)
    generations.save_interpolation(
        os.path.join(opt.save, "interp", str(step) + '-'),
        model, volatile_sequence)
    model.train()


def save_checkpoint(step, state):
    save_dict = {
        'model': model,
        'opt': vars(opt),
        'i': step,
    }
    torch.save(save_dict, opt.save + '/model.t7')


bookkeeper = Bookkeeper(i, reset_state({}), update_reducer)
bookkeeper.every(opt.print_every, make_log)
bookkeeper.every(opt.save_every, save_checkpoint)


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()

while i < opt.max_steps:
    i += opt.batch_size
    sequence = next(training_batch_generator)

    # ---- train the autoencoder -----
    output = model(sequence)
    generated_sequence = output['generations']
    loss = output['loss']

    model.zero_grad()
    loss.backward()
    optimizer.step()

    bookkeeper.update(i, output)

    if i >= opt.max_steps:
        break
