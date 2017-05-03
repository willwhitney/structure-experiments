import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.serialization import load_lua

import scipy.misc
import imageio
import glob
import os
import math
import random

def fps(path):
    if path.find('1fps') >= 0:
        return 1
    elif path.find('4fps') >= 0:
        return 4
    else:
        return 2

def get_videos(directory):
    if directory[-4:].lower() == '.mp4':
        filenames = [directory]
    else:
        filenames = glob.glob("{}/*.MP4".format(directory))
    return filenames

class Video(list):
    def __init__(self, path, image_size):
        super(Video, self).__init__()
        self.path = path
        self.name = os.path.basename(path)
        self.image_size = image_size
        self.fps = fps(path)

        v = imageio.get_reader(path, 'avbin')
        for frame in v:
            resized = scipy.misc.imresize(frame, self.image_size[1:])
            resized = torch.from_numpy(resized).float()
            resized.transpose_(1, 2).transpose_(0, 1).div_(256)
            self.append(resized)
        v.close()

class VideoChunk(list):
    def __init__(self, frames, framerate):
        super(VideoChunk, self).__init__()
        self.extend(frames)
        self.fps = framerate

def make_split_datasets(directory, seq_len, framerate,
                        image_width=128, chunk_length=1000):
    filenames = get_videos(directory)

    videos = []
    for fname in filenames:
        if fps(fname) >= framerate:
            print(fname)
            v = Video(fname, [3, image_width, image_width])
            videos.append(v)

    chunks = []
    for v in videos:
        for i in range(0, len(v), chunk_length):
            chunk = VideoChunk(v[i : i + chunk_length], framerate)
            chunks.append(chunk)

    train_frac = 0.9
    test_frac = 1 - train_frac
    test_chunk_indices = random.sample(list(range(len(chunks))),
                                       math.ceil(len(chunks) * test_frac))
    test_chunk_indices = set(test_chunk_indices)
    train_chunk_indices = set(list(range(len(chunks))))
    train_chunk_indices = train_chunk_indices - test_chunk_indices

    train_chunks = [chunks[i] for i in train_chunk_indices]
    test_chunks = [chunks[i] for i in test_chunk_indices]

    train_dataset = ChunkData(train_chunks, seq_len, framerate, image_width)
    test_dataset = ChunkData(test_chunks, seq_len, framerate, image_width)
    return train_dataset, test_dataset

class VideoData(Dataset):
    def __init__(self, directory, seq_len, framerate, image_width=128):
        self.seq_len = seq_len
        self.framerate = framerate
        self.filenames = get_videos(directory)
        self.image_size = [3, image_width, image_width]

        self.loaded = {}

        self.videos = []
        for fname in self.filenames:
            if fps(fname) >= framerate:
                print(fname)
                v = Video(fname, self.image_size)
                self.videos.append(v)

    def _end_padding(self, v):
        return math.ceil(self.seq_len * v.fps / self.framerate)

    def __getitem__(self, i):
        # do some bookkeeping to find the video that contains i
        current_count = 0
        correct_v = None
        for v in self.videos:
            end_padding = self._end_padding(v)
            if current_count + len(v) - end_padding > i:
                correct_v = v
                break
            else:
                current_count += len(v) - end_padding
        start_frame = i - current_count

        seq = torch.Tensor(self.seq_len, *self.image_size)
        for t in range(self.seq_len):
            skipped_t = t * math.ceil(v.fps / self.framerate)
            frame = v[start_frame + skipped_t]
            seq[t].copy_(frame)
        return seq

    # length is the number of indices in the video that have at least
    # seq_len * fps frames after them (i.e., complete sequences at 1 fps)
    def __len__(self):
        sequences = 0
        for v in self.videos:
            sequences += len(v) - self._end_padding(v)
        return sequences

class ChunkData(VideoData):
    def __init__(self, chunks, seq_len, framerate,
                 image_width=128):
        self.seq_len = seq_len
        self.framerate = framerate
        self.image_size = [3, image_width, image_width]
        self.videos = chunks

        self.loaded = {}

# train, test = make_split_datasets('.', 5,
#                                   framerate=2,
#                                   image_width=128,
#                                   chunk_length=50)

# v = Video('/speedy/data/urban/1fps_1.MP4', [3, 128, 128])
# from util import *
# v = Video('output.MP4', [3, 128, 128])

# data = VideoData('/speedy/data/urban', 5)
# for i, d in enumerate(data):
#     print(i)

# loader = DataLoader(data, num_workers = 0, batch_size = 32, shuffle = True)
