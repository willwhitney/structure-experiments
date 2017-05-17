import torch
import socket
import argparse
import json
import glob
import os

from util import *

hostname = socket.gethostname()
# if socket.gethostname() == 'zaan':
#     dtype = torch.cuda.FloatTensor
# else:
#     dtype = torch.FloatTensor

dtype = torch.cuda.FloatTensor
# batch_size = 64

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=get_gpu())
parser.add_argument('--sanity', action="store_true")

parser.add_argument('--load', default=None)
parser.add_argument('--use_loaded_opt', action="store_true")
parser.add_argument('--resume', action="store_true")

parser.add_argument('--data', default='urban/5th_ave')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--fps', default=4, type=int)
parser.add_argument('--seq_len', default=5, type=int)
parser.add_argument('--data_sparsity', default=1, type=int)
parser.add_argument('--motion_weight', default=0, type=int)

parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--no_lr_decay', action="store_true")
parser.add_argument('--no_sgld', action="store_true")
parser.add_argument('--activation', default="tanh")
parser.add_argument('--no_kl_annealing', action="store_true")
parser.add_argument('--kl_anneal_end', default=3e6, type=float)
parser.add_argument('--kl_weight', default=1, type=int)
parser.add_argument('--output_var', default=0.01, type=float)
parser.add_argument('--latents', default=3, type=int)
parser.add_argument('--latent_dim', default=25, type=int)
parser.add_argument('--trans_layers', default=4, type=int)

parser.add_argument('--game', default='freeway')

parser.add_argument('--colors', default='random',
                    help="color of bouncing balls. white | vary | random")
parser.add_argument('--balls', default=1, type=int,
                    help="number of balls in the environment")

parser.add_argument('--image_width', default=128, type=int)
opt = parser.parse_args()

batch_size = opt.batch_size

def make_result_folder(location):
    if not os.path.exists(location):
        os.makedirs(location)
    else:
        filelist = glob.glob(location + "/*")
        if len(filelist) > 0:
            clear = query_yes_no(
                "This network name is already in use. "
                "Continuing will delete all of the files in the directory.\n"
                "Files: \n" + "\n".join(filelist) + "\n\n"
                "Continue?")
            if not clear:
                print("Not deleting anything. Quitting instead.")
                exit()
            for f in filelist:
                os.remove(f)

def write_options(location):
    with open(location + "/opt.json", 'w') as f:
        serial_opt = json.dumps(vars(opt), indent=4, sort_keys=True)
        print(serial_opt)
        f.write(serial_opt)
        f.flush()
