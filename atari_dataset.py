import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.serialization import load_lua

import scipy.misc
import glob
import socket

class AtariData(Dataset):
    def __init__(self, game, mode, seq_len, image_width):
        self.seq_len = seq_len
        if socket.gethostname() == 'zaan':
            data_path = "/speedy/data/atari"
        else:
            data_path = "/misc/vlgscratch2/FergusGroup/wwhitney/data/atari"
        self.filenames = glob.glob(
            "{}/{}/{}/images_*".format(
                data_path, game, mode
            ))
        self.image_size = [3, image_width, image_width]

        self.loaded = {}

    # def __getitem__(self, i):
    #     return torch.rand(5, 3, 64, 64).type(self.dtype)

    def __getitem__(self, i):
        if i in self.loaded:
            # print('cache!')
            return self.loaded[i]
        raw = load_lua(self.filenames[i])
        result = torch.Tensor(self.seq_len, *self.image_size)
        for t in range(self.seq_len):
            skipped_t = t * 5
            np_resized = scipy.misc.imresize(raw[skipped_t].numpy(),
                                             self.image_size[1:])
            resized = torch.from_numpy(np_resized)
            resized.transpose_(1, 2).transpose_(0, 1)
            result[t].copy_(resized)
        result = result / 256
        self.loaded[i] = result
        # print(self.loaded.keys())
        return result

    def __len__(self):
        # return 10
        return len(self.filenames)

# d = AtariData('freeway', 'train', 5, torch.cuda.FloatTensor)
# l = DataLoader(d, num_workers=4, batch_size=32, shuffle=True)
#
# single = d[0]
# batch = None
# for b in l:
#     batch = b
#     break
