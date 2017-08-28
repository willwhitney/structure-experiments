import torch
import socket
import argparse
import json
import glob
import os

from atari_dataset import AtariData
from video_dataset import *
from env import *
from util import *

hostname = socket.gethostname()
if socket.gethostname().find('touchy') >= 0:
    dtype = torch.FloatTensor
else:
    dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=get_gpu())
parser.add_argument('--sanity', action="store_true")

parser.add_argument('--commit', dest='commit', action='store_true')
parser.add_argument('--no-commit', dest='commit', action='store_false')
parser.set_defaults(commit=True)

parser.add_argument('--git', dest='git', action='store_true')
parser.add_argument('--no-git', dest='git', action='store_false')
parser.set_defaults(git=True)

parser.add_argument('--kl-anneal', dest='kl_anneal', action='store_true')
parser.add_argument('--no-kl-anneal', dest='kl_anneal', action='store_false')
parser.set_defaults(kl_anneal=True)

parser.add_argument('--load', default=None)
parser.add_argument('--use-loaded-opt', action="store_true")
parser.add_argument('--resume', action="store_true")

parser.add_argument('--print-every', default=200000, type=int)
parser.add_argument('--max-steps', default=5e8, type=int)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--data', default='urban/5th_ave')
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--fps', default=4, type=int)
parser.add_argument('--seq-len', default=5, type=int)
parser.add_argument('--data-sparsity', default=1, type=int)
parser.add_argument('--motion-weight', default=0, type=int)

parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--no-lr_decay', action="store_true")
parser.add_argument('--no-sgld', action="store_true")
parser.add_argument('--activation', default="tanh")
parser.add_argument('--kl-anneal-end', default=3e6, type=float)
parser.add_argument('--kl-weight', default=1, type=int)
parser.add_argument('--output-var', default=0.01, type=float)
parser.add_argument('--latents', default=3, type=int)
parser.add_argument('--latent-dim', default=25, type=int)
parser.add_argument('--trans-layers', default=4, type=int)
parser.add_argument('--tiny', action="store_true")

parser.add_argument('--game', default='freeway')

parser.add_argument('--colors', default='random',
                    help="color of bouncing balls. white | vary | random")
parser.add_argument('--balls', default=1, type=int,
                    help="number of balls in the environment")

parser.add_argument('--image-width', default=128, type=int)
parser.add_argument('--channels', default=3, type=int)
opt = parser.parse_args()

# batch_size = opt.batch_size
