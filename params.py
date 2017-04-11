import torch
import socket
import argparse

from util import *

if socket.gethostname() == 'zaan':
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=get_gpu())
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--lr_decay', action="store_true")
parser.add_argument('--sgld', action="store_true")
parser.add_argument('--activation', default="tanh")
parser.add_argument('--no_kl_annealing', action="store_true")

opt = parser.parse_args()
opt.save = 'results/' + opt.name
