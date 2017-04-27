import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.serialization import load_lua

import scipy.misc
import imageio
import glob
import os

class Video(list):
    def __init__(self, path, image_size):
        super(Video, self).__init__()
        self.path = path
        self.name = os.path.basename(path)
        self.image_size = image_size
        self.fps = 1 if self.name.find('1fps') >= 0 else 2

        v = imageio.get_reader(path, 'avbin')
        for frame in v:
            resized = scipy.misc.imresize(frame, self.image_size[1:])
            resized = torch.from_numpy(resized).float()
            resized.transpose_(1, 2).transpose_(0, 1).div_(256)
            self.append(resized)
        v.close()

class VideoData(Dataset):
    def __init__(self, directory, seq_len):
        self.seq_len = seq_len
        self.filenames = glob.glob("{}/*.MP4".format(directory))
        self.image_size = [3, 128, 128]

        self.loaded = {}

        self.videos = []
        for fname in self.filenames:
            v = imageio.get_reader(fname, 'avbin')

            # vlist = []
            # for frame in v:
            #     resized = scipy.misc.imresize(frame, self.image_size[1:])
            #     resized = torch.from_numpy(resized).float()
            #     resized.transpose_(1, 2).transpose_(0, 1)
            #     vlist.append(resized)
            # self.videos.append(vlist)
            # v.close()

            v = Video(fname, self.image_size)
            self.videos.append(v)

    def __getitem__(self, i):
        # do some bookkeeping to find the video that contains i
        current_count = 0
        v_index = 0
        while current_count + len(self.videos[v_index]) - self.seq_len < i:
            current_count += len(self.videos[v_index])
            v_index += 1
        v = self.videos[v_index]
        start_frame = i - current_count

        seq = torch.Tensor(self.seq_len, *self.image_size)
        for t in range(self.seq_len):
            skipped_t = t * v.fps

            # np_resized = scipy.misc.imresize(v[start_frame + skipped_t],
            #                                  self.image_size[1:])
            # resized = torch.from_numpy(np_resized).float()
            # resized.transpose_(1, 2).transpose_(0, 1)
            # seq[t].copy_(resized)
            frame = v[start_frame + skipped_t]
            seq[t].copy_(frame)
        return seq

    # length is the number of indices in the video that have at least
    # seq_len * fps frames after them (i.e., complete sequences at 1 fps)
    def __len__(self):
        sequences = 0
        for v in self.videos:
            sequences += len(v) - self.seq_len * v.fps
        return sequences

# v = Video('/speedy/data/urban/1fps_1.MP4', [3, 128, 128])
# from util import *
# v = Video('output.MP4', [3, 128, 128])

# data = VideoData('.', 5)
# DataLoader(data, num_workers = 0, batch_size = 32, shuffle = True)
# d = data[17]
#
# show(d0[0])
