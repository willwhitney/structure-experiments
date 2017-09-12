import torch.utils.data as data
from moving_mnist import data_handler as dh
import h5py
import progressbar as pb
import torch

class MovingMNIST(data.Dataset):
    def __init__(self, train=True, seq_len=20, image_size=64, colored=False):
        if colored:
            handler = dh.ColoredBouncingMNISTDataHandler
        else:
            handler = dh.BouncingMNISTDataHandler

        # tiny = True
        tiny = False
        if train == True:
            self.data_handler = handler(seq_len, image_size)
            self.data_size = 64 * pow(2, 2) if tiny else 64 * pow(2, 12)
            
        else:
            self.data_handler = handler(seq_len, image_size)
            self.data_size = 64 * pow(2, 2) if tiny else 64 * pow(2, 5)

        pbar = pb.ProgressBar()

        self.data = []

        print("Generating dataset:")
        for i in pbar(range(self.data_size)):
            self.data.append(
                    torch.from_numpy(self.data_handler.GetItem()) / 255)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data_size
