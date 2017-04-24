import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.serialization import load_lua

import scipy.misc
import imageio
import glob

class VideoData(Dataset):
    def __init__(self, directory, seq_len):
        self.seq_len = seq_len
        self.filenames = glob.glob("{}/*.MP4".format(directory))
        self.image_size = [3, 64, 64]

        self.loaded = {}

        self.videos = []
        for fname in self.filenames:
            vlist = []
            v = imageio.get_reader(fname, 'avbin')
            for frame in v:
                resized = scipy.misc.imresize(frame, self.image_size[1:])
                resized = torch.from_numpy(resized).float()
                resized.transpose_(1, 2).transpose_(0, 1)
                vlist.append(resized)
            self.videos.append(vlist)


    def __getitem__(self, i):
        # if i in self.loaded:
            # return self.loaded[i]

        current_count = 0
        v_index = 0
        while current_count + len(self.videos[v_index]) - self.seq_len < i:
            current_count += len(self.videos[v_index])
            v_index += 1
        v = self.videos[v_index]
        start_frame = i - current_count

        seq = torch.Tensor(self.seq_len, *self.image_size)
        for t in range(self.seq_len):
            skipped_t = t * 1

            # np_resized = scipy.misc.imresize(v[start_frame + skipped_t],
            #                                  self.image_size[1:])
            # resized = torch.from_numpy(np_resized).float()
            # resized.transpose_(1, 2).transpose_(0, 1)
            # seq[t].copy_(resized)
            frame = v[start_frame + skipped_t]
            seq[t].copy_(frame)
        seq = seq / 256
        # self.loaded[i] = seq
        # print(self.loaded.keys())
        return seq

    def __len__(self):
        return sum([len(v) - self.seq_len for v in self.videos])


# data = VideoData('.', 5)
# DataLoader(data, num_workers = 0, batch_size = 32, shuffle = True)
# d0 = data[0]
#
# from util import *
# show(d0[0])
