import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.serialization import load_lua

import scipy.misc
import glob

class AtariData(Dataset):
    def __init__(self, game, mode, seq_len):
        self.seq_len = seq_len
        self.filenames = glob.glob(
            "/speedy/data/atari/{}/{}/images_*".format(
                game, mode
            ))
        self.image_size = [3, 64, 64]

    # def __getitem__(self, i):
    #     return torch.rand(5, 3, 64, 64).type(self.dtype)

    def __getitem__(self, i):
        raw = load_lua(self.filenames[i])
        result = torch.Tensor(self.seq_len, *self.image_size)
        for t in range(self.seq_len):
            np_resized = scipy.misc.imresize(raw[t].numpy(),
                                             self.image_size[1:])
            resized = torch.from_numpy(np_resized)
            resized.transpose_(1, 2).transpose_(0,1)
            result[t].copy_(resized)
        return result / 256

    def __len__(self):
        return len(self.filenames)

# d = AtariData('freeway', 'train', 5, torch.cuda.FloatTensor)
# l = DataLoader(d, num_workers=4, batch_size=32, shuffle=True)
#
# single = d[0]
# batch = None
# for b in l:
#     batch = b
#     break
