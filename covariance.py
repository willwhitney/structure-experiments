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

import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import sklearn.covariance

from util import ensure_path_exists


@ensure_path_exists
def construct_covariance(savedir, list_of_samples, latent_dim, label):
    z_dim = list_of_samples[0].size(0)
    n_latents = z_dim // latent_dim

    tensor_of_samples = torch.stack(list_of_samples, 0)
    data = tensor_of_samples.cpu().numpy()

    df = pd.DataFrame(data)
    corr_df = df.corr()
    corr = np.array(corr_df)

    fig = plt.figure()
    hm = seaborn.heatmap(corr,
                         xticklabels=latent_dim,
                         yticklabels=latent_dim)

    for i in range(1, n_latents):
        location = i * latent_dim
        plt.plot([0, z_dim], [location, location],
                 color='black', linewidth=0.5)
        plt.plot([location, location], [0, z_dim],
                 color='black', linewidth=0.5)
    name = "{}corr_{}.pdf".format(savedir, label)
    plt.savefig(name)
    plt.close(fig)

    fig = plt.figure()
    cov = np.array(df.cov())
    hm = seaborn.heatmap(cov,
                         xticklabels=latent_dim,
                         yticklabels=latent_dim)

    for i in range(1, n_latents):
        location = i * latent_dim
        plt.plot([0, z_dim], [location, location],
                 color='black', linewidth=0.5)
        plt.plot([location, location], [0, z_dim],
                 color='black', linewidth=0.5)
    name = "{}cov_{}.pdf".format(savedir, label)
    plt.savefig(name)
    plt.close(fig)


if __name__ == "__main__":
    # dataset = VideoData('.', 5, framerate=2)
    # dataset = VideoData('/speedy/data/urban', 5, framerate=2)
    if hostname == 'zaan':
        data_path = '/speedy/data/urban'
    else:
        data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/urban'
    train_data, test_data = make_split_datasets(
        data_path, 5, framerate=2, image_width=opt.image_width,
        chunk_length=20, train_frac=0.5)
    # dataset = VideoData('/speedy/data/urban/tuesday_4fps.MP4', 5, framerate=2)
    train_loader = DataLoader(train_data,
                              num_workers=0,
                              batch_size=opt.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_data,
                             num_workers=0,
                             batch_size=opt.batch_size,
                             shuffle=True)

    checkpoint = torch.load('results/' + opt.load + '/model.t7')
    model = checkpoint['model'].type(dtype)
    cp_opt = checkpoint['opt']
    setattrs(opt, cp_opt, exceptions=['name', 'load', 'sanity'])

    construct_covariance('results/' + opt.load, model, train_loader, 10000,
                         label="chunk50_train")
    construct_covariance('results/' + opt.load, model, test_loader, 10000,
                         label="chunk50_test")
