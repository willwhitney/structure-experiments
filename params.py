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
# if socket.gethostname() == 'zaan':
#     dtype = torch.cuda.FloatTensor
# else:
#     dtype = torch.FloatTensor

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--name', default=get_gpu())
parser.add_argument('--sanity', action="store_true")

parser.add_argument('--commit', dest='commit', action='store_true')
parser.add_argument('--no-commit', dest='commit', action='store_false')
parser.set_defaults(commit=True)

parser.add_argument('--git', dest='git', action='store_true')
parser.add_argument('--no-git', dest='git', action='store_false')
parser.set_defaults(git=True)

parser.add_argument('--kl-anneal', dest='kl-anneal', action='store_true')
parser.add_argument('--no-kl-anneal', dest='kl-anneal', action='store_false')
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
opt = parser.parse_args()

# batch_size = opt.batch_size

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

    # copy over the old results if we're resuming
    if opt.resume:
        shutil.copyfile(cp_opt['save'] + '/results.csv',
                        opt.save + '/results.csv')

def write_options(location):
    with open(location + "/opt.json", 'w') as f:
        serial_opt = json.dumps(vars(opt), indent=4, sort_keys=True)
        print(serial_opt)
        f.write(serial_opt)
        f.flush()

def load_checkpoint():
    checkpoint = torch.load('results/' + opt.load + '/model.t7')
    model = checkpoint['model'].type(dtype)
    i = checkpoint['i']
    cp_opt = checkpoint['opt']

    # we're strictly trying to pick up where we left off
    # load everything just as it was (but rename)
    if opt.resume:
        setattrs(opt, cp_opt, exceptions=['load', 'resume', 'use_loaded_opt'])
        opt.name = cp_opt['name'] + '_'

    # if we want to use the options from the checkpoint, load them in
    # (skip the ones that don't make sense to load)
    if opt.use_loaded_opt:
        setattrs(opt, cp_opt, exceptions=[
            'name', 'load', 'sanity', 'resume', 'use_loaded_opt'
        ])
    return i, model

def load_dataset():
    if opt.resume:
        data_path = 'results/' + opt.load + '/dataset.t7'
        print("Loading stored dataset from {}".format(data_path))
        data_checkpoint = torch.load(data_path)
        train_data = data_checkpoint['train_data']
        test_data = data_checkpoint['test_data']

    else:
        if hostname == 'zaan':
            data_path = '/speedy/data/' + opt.data
            # data_path = '/speedy/data/urban/5th_ave'
        else:
            data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/' + opt.data
            # data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/urban/5th_ave'

        # 'urban' datasets are in-memory stores
        if data_path.find('urban') >= 0:
            if not data_path[-3:] == '.t7':
                data_path = data_path + '/dataset.t7'

            print("Loading stored dataset from {}".format(data_path))
            data_checkpoint = torch.load(data_path)
            train_data = data_checkpoint['train_data']
            test_data = data_checkpoint['test_data']

            train_data.seq_len = opt.seq_len
            test_data.seq_len = opt.seq_len

            load_workers = 0

        elif opt.data == 'atari':
            train_data = AtariData(
                opt.game, 'train', opt.seq_len, opt.image_width)
            test_data = AtariData(
                opt.game, 'test', opt.seq_len, opt.image_width)
            load_workers = 0

        elif opt.data == 'balls':
            train_data = BounceData(
                opt.seq_len, opt.balls, opt.colors, opt.image_width)
            test_data = BounceData(
                opt.seq_len, opt.balls, opt.colors, opt.image_width)
            load_workers = 0

        elif opt.data == '1d_balls':
            train_data = HorizontalBounceData(
                opt.seq_len, opt.balls, opt.colors, opt.image_width)
            test_data = HorizontalBounceData(
                opt.seq_len, opt.balls, opt.colors, opt.image_width)
            load_workers = 0

        # other video datasets are big and stored as chunks
        else:
            if hostname != 'zaan':
                scratch_path = '/scratch/wwhitney/' + opt.data
                vlg_path = '/misc/vlgscratch4/FergusGroup/wwhitney/' + opt.data

                data_path = vlg_path
                # if os.path.exists(scratch_path):
                #     data_path = scratch_path
                # else:
                #     data_path = vlg_path

            print("Loading stored dataset from {}".format(data_path))
            train_data, test_data = load_disk_backed_data(data_path)

            if opt.data_sparsity > 1:
                train_data.videos = [train_data.videos[i]
                                     for i in range(len(train_data.videos))
                                     if i % opt.data_sparsity == 0]
            load_workers = 4

            train_data.framerate = opt.fps
            test_data.framerate = opt.fps

            train_data.seq_len = opt.seq_len
            test_data.seq_len = opt.seq_len
    return train_data, test_data, load_workers
