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

from util import *

from modules import *
from models import *
from env import *
from params import *
from generations import *
from atari_dataset import AtariData
from video_dataset import *

import sklearn.covariance

def construct_covariance(savedir, model, loader, n, label=""):
    data = []
    for i, batch in enumerate(loader):
        data.append(batch)
        if i * loader.batch_size > n:
            break

    KL = GaussianKLD().type(dtype)
    LL = GaussianLL().type(dtype)

    def sequence_input(seq):
        return [Variable(x.type(dtype)) for x in seq]

    zs = []

    for i in range(len(data)):
        x = data[i]
        x.transpose_(0, 1)
        x = sequence_input(list(x))

        for a in x:
            a.volatile = True

        inferred_z_post = model.first_inference(x[0])
        cat_prior = model.z1_prior

        seq_zs = []
        for t in range(len(x)):
            seq_zs.append(inferred_z_post[0])
            z_sample = sample(inferred_z_post)
            # gen_dist = model.generator(z_sample)

            if t < len(x) - 1:
                cat_prior = model.predict_latent(z_sample)
                inferred_z_post = model.inference(x[t+1],
                                                  cat_prior[0])
        zs.append(seq_zs)

    paired_zs = []
    for seq_zs in zs:
        t = random.randint(0, len(seq_zs) - 2)
        # for t in range(len(seq_zs) - 1):
        z = seq_zs[t]
        z_n = seq_zs[t+1]
        for first, second in zip(z, z_n):
            catted = torch.cat([first, second], 0)
            paired_zs.append(catted)
    zs = None

    paired_zs = [z.data for z in paired_zs]
    paired_zs = torch.stack(paired_zs, 1)
    paired_zs = paired_zs.cpu()
    paired_zs.transpose_(0, 1)
    paired_zs = paired_zs.numpy()

    z_dim = model.n_latents * model.hidden_dim

    import matplotlib.pyplot as plt
    import seaborn
    import pandas as pd

    df = pd.DataFrame(paired_zs)
    corr_df = df.corr()
    corr_np = np.array(corr_df)

    # try to free things up?
    paired_zs = None

    fig = plt.figure()
    corr = corr_np[:z_dim, :z_dim]
    hm = seaborn.heatmap(corr,
                         xticklabels=model.hidden_dim,
                         yticklabels=model.hidden_dim)

    for i in range(1, model.n_latents):
        location = i * model.hidden_dim
        plt.plot([0, z_dim], [location, location], color='black', linewidth=0.5)
        plt.plot([location, location], [0, z_dim], color='black', linewidth=0.5)
    name = "{}/autocorr_{}.pdf".format(savedir, label)
    plt.savefig(name)
    plt.close(fig)


    fig = plt.figure()
    corr = corr_np[:z_dim, z_dim:]
    hm = seaborn.heatmap(corr,
                         xticklabels=model.hidden_dim,
                         yticklabels=model.hidden_dim)

    for i in range(1, model.n_latents):
        location = i * model.hidden_dim
        plt.plot([0, z_dim], [location, location], color='black', linewidth=0.5)
        plt.plot([location, location], [0, z_dim], color='black', linewidth=0.5)
    # plt.savefig('results/' + name + '/corr.pdf')
    name = "{}/corr_{}.pdf".format(savedir, label)
    plt.savefig(name)
    plt.close(fig)

    fig = plt.figure()
    cov = np.array(df.cov())
    # cov = sklearn.covariance.empirical_covariance(paired_zs)
    hm = seaborn.heatmap(cov[:z_dim, z_dim:],
                         xticklabels=model.hidden_dim,
                         yticklabels=model.hidden_dim)

    for i in range(1, model.n_latents):
        location = i * model.hidden_dim
        plt.plot([0, z_dim], [location, location], color='black', linewidth=0.5)
        plt.plot([location, location], [0, z_dim], color='black', linewidth=0.5)
    # plt.savefig('results/' + name + '/cov.pdf')
    name = "{}/cov_{}.pdf".format(savedir, label)
    plt.savefig(name)
    plt.close(fig)


if __name__ == "__main__":
    # dataset = VideoData('.', 5, framerate=2)
    # dataset = VideoData('/speedy/data/urban', 5, framerate=2)
    train_data, test_data = make_split_datasets(
        '/speedy/data/urban', 5, framerate=2, image_width=opt.image_width,
        chunk_length=20, train_frac=0.5)
    # dataset = VideoData('/speedy/data/urban/tuesday_4fps.MP4', 5, framerate=2)
    train_loader = DataLoader(train_data,
                        num_workers=0,
                        batch_size=batch_size,
                        shuffle=True)
    test_loader = DataLoader(test_data,
                        num_workers=0,
                        batch_size=batch_size,
                        shuffle=True)

    checkpoint = torch.load('results/' + opt.load + '/model.t7')
    model = checkpoint['model'].type(dtype)
    cp_opt = checkpoint['opt']
    setattrs(opt, cp_opt, exceptions=['name', 'load', 'sanity'])

    construct_covariance('results/' + opt.load, model, train_loader, 10000,
                         label="chunk50_train")
    construct_covariance('results/' + opt.load, model, test_loader, 10000,
                         label="chunk50_test")
