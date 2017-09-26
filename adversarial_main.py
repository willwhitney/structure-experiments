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
    i, autoencoder, adversary = load_adversarial_checkpoint(opt, dtype)
else:
    i = 0
    adversary = IndependenceAdversary(opt.latents, opt.latent_dim).type(dtype)
    autoencoder = IndependentAutoencoder(
        opt.latents, opt.latent_dim,
        opt.image_width, adversary,
        inference=DeterministicTinyDCGANFirstInference,
        generator=DeterministicTinyDCGANGenerator).type(dtype)

opt.save = 'results/' + opt.name
print(adversary)
print(autoencoder)

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
logging.debug(("step,autoencoder_loss,reconstruction_loss,adversarial_loss,"
               "adversary_loss,adversary_true_loss,adversary_false_loss,"
               "true_outputs,false_outputs,"
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
autoencoder_optimizer = optim.Adam(
    autoencoder.parameters(),
    lr=opt.lr)

adversary_optimizer = optim.Adam(
    adversary.parameters(),
    lr=opt.lr)


# def lr_lambda(epoch): return opt.lr_decay ** (i / 320000)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------- set up bookkeeper and its functions ---------------
sequence, reconstruction = None, None


def reset_state(state):
    state['autoencoder_loss'] = 0
    state['reconstruction_loss'] = 0
    state['adversarial_loss'] = 0
    state['adversary_loss'] = 0
    state['true_loss'] = 0
    state['false_loss'] = 0
    state['true_outputs'] = 0
    state['false_outputs'] = 0
    state['progress'] = progressbar.ProgressBar(max_value=opt.print_every)
    return state


def update_reducer(step, state, updates):
    state['progress'].update(step % opt.print_every)

    # seq_divergence, seq_prior_div, seq_trans_div = updates['divergences']
    state['autoencoder_loss'] += updates['autoencoder_loss'].data[0]
    state['reconstruction_loss'] += updates['reconstruction_loss'].data[0]
    state['adversarial_loss'] += updates['adversarial_loss'].data[0]
    state['adversary_loss'] += updates['adversary_loss'].data[0]
    state['true_loss'] += updates['true_loss'].data[0]
    state['false_loss'] += updates['false_loss'].data[0]
    state['true_outputs'] += updates['true_outputs'].data
    state['false_outputs'] += updates['false_outputs'].data


def make_log(step, state):
    state['progress'].finish()
    clear_progressbar()

    elapsed_time = state['progress'].end_time - state['progress'].start_time
    elapsed_seconds = elapsed_time.total_seconds()
    batches = opt.print_every / opt.batch_size
    log_values = (step,
                    state['autoencoder_loss'] / batches,
                    state['reconstruction_loss'] / batches,
                    state['adversarial_loss'] / batches,
                    state['adversary_loss'] / batches,
                    state['true_loss'] / batches,
                    state['false_loss'] / batches,
                    state['true_outputs'].mean() / batches,
                    state['false_outputs'].mean() / batches,
                    elapsed_seconds / opt.print_every * 1000)

    print(("Step: {:8d}, AE loss: {:8.3f}, Recon: {:8.3f}, Advers: {:8.3f}, "
            "Adversary loss: {:8.3f}, T loss: {:8.3f}, F loss: {:8.3f}, "
            "T outputs: {:6.3f}, F outputs: {:6.3f}, "
            "ms/seq: {:6.2f}").format(*log_values))

    # make list of n copies of format string, then format
    format_string = ",".join(["{:.8e}"] * len(log_values))
    logging.debug(format_string.format(*log_values))
    reset_state(state)

    for x in sequence:
        x.volatile = True
    generations.save_paired_sequence(
        os.path.join(opt.save, "reconstruction", str(step) + '-'),
        sequence[:2], reconstruction)
    generations.save_single_replacement(
        os.path.join(opt.save, "ind_replace", str(step) + '-'), 
        autoencoder, sequence)


def save_checkpoint(step, state):
    save_dict = {
        'autoencoder': autoencoder,
        'adversary': adversary,
        'opt': vars(opt),
        'i': step,
    }
    torch.save(save_dict, opt.save + '/model.t7')


# def make_covariance(step, state):
#     if opt.seq_len > 1:
#         construct_covariance(opt.save + '/covariance/',
#                              model, train_loader, 10,
#                              label="train_" + str(step))
#         construct_covariance(opt.save + '/covariance/',
#                              model, test_loader, 10,
#                              label="test_" + str(step))


bookkeeper = Bookkeeper(i, reset_state({}), update_reducer)
bookkeeper.every(opt.print_every, make_log)
bookkeeper.every(opt.save_every, save_checkpoint)
# bookkeeper.every(opt.cov_every, make_covariance)

last_autoencoder_output = None
while i < opt.max_steps:
    for sequence in train_loader:
        i += opt.batch_size

        sequence = normalize_data(opt, dtype, sequence)
        if last_autoencoder_output is None:
            last_autoencoder_output = autoencoder(sequence[:2])
            continue


        # ---- train the autoencoder -----
        autoencoder_output = autoencoder(sequence[:2])
        reconstruction = autoencoder_output['reconstruction']
        reconstruction = [reconstruction[:opt.batch_size],
                          reconstruction[opt.batch_size:]]
        recon_loss = autoencoder_output['reconstruction_loss']
        adversarial_loss = autoencoder_output['adversarial_loss']
        autoencoder_loss = recon_loss + 0.5 * adversarial_loss

        autoencoder.zero_grad()
        autoencoder_loss.backward()
        autoencoder_optimizer.step()


        adversary_input_latents0 = Variable(
            torch.ones(opt.batch_size, 
            opt.latents * opt.latent_dim)).type(dtype)
        adversary_input_latents1 = Variable(
            torch.zeros(opt.batch_size, 
            opt.latents * opt.latent_dim)).type(dtype)
        adversary_input_latentsbar = Variable(
            torch.ones(opt.batch_size, 
            opt.latents * opt.latent_dim) * 0.5).type(dtype)
        # adversary_output = adversary(autoencoder_output['latents0'],
        #                              autoencoder_output['latents1'],
        #                              adversary_input_latentsbar)
        # adversary_output = adversary(adversary_input_latents0,
        #                              adversary_input_latents1,
        #                              adversary_input_latentsbar)

        # ---- train the adversary -----
        adversary_output = adversary(autoencoder_output['latents0'],
                                     autoencoder_output['latents1'],
                                     last_autoencoder_output['latents0'])

        true_loss = adversary_output['true_loss']
        false_loss = adversary_output['false_loss']
        adversary_loss = true_loss + false_loss

        adversary.zero_grad()
        adversary_loss.backward()
        adversary_optimizer.step()


        bookkeeper.update(i, {
            'autoencoder_loss': autoencoder_loss,
            'adversary_loss': adversary_loss,
            **autoencoder_output, **adversary_output
        })

        if i >= opt.max_steps:
            break

        last_autoencoder_output = autoencoder_output
