import torch
from torch.utils.data import Dataset, DataLoader
# from torch.utils.serialization import load_lua

import scipy.misc
import imageio
import glob
import os
import math
import random
import uuid

def fps(path):
    if path.find('1fps') >= 0:
        return 1
    elif path.find('2fps') >= 0:
        return 2
    elif path.find('4fps') >= 0:
        return 4
    else:
        return 30

def get_videos(directory):
    video_formats = ['MP4', 'mp4', 'avi']
    filenames = []
    for form in video_formats:
        if directory[-4:].lower() == '.' + form.lower():
            filenames.append(directory)
            break
        else:
            form_filenames = glob.glob("{}/*.{}".format(directory, form))
            filenames.extend(form_filenames)
            # form_filenames = glob.glob("{}/camera3*.{}".format(directory, form))
            # filenames.extend(form_filenames)
            # form_filenames = glob.glob("{}/camera5*.{}".format(directory, form))
            # filenames.extend(form_filenames)
    return filenames

class Video(list):
    def __init__(self, path, image_size):
        super(Video, self).__init__()
        self.path = path
        self.name = os.path.basename(path)
        self.image_size = image_size
        self.fps = fps(path)

        v = imageio.get_reader(path, 'ffmpeg')
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

def load_wrap(f):
    def wrapper(self, *args, **kwargs):
        # print("Running the wrapper!")
        # print(self)
        # import ipdb; ipdb.set_trace()
        self.frames = torch.load(self.path)
        result = f(self, *args, **kwargs)
        self.frames = None
        return result
    return wrapper

class DiskVideoChunk():
    def __init__(self, frames, framerate, storage_path):
        super(DiskVideoChunk, self).__init__()
        self.path = storage_path
        self.fps = framerate
        self.length = len(frames)
        torch.save(frames, self.path)

        # they'll be loaded on the fly later
        self.frames = None

    @load_wrap
    def __getitem__(self, *args, **kwargs):
        return self.frames.__getitem__(*args, **kwargs)

    @load_wrap
    def __iter__(self, *args, **kwargs):
        return self.frames.__iter__(*args, **kwargs)

    def __len__(self):
        return self.length


def make_split_datasets(directory, seq_len, framerate,
                        image_width=128, chunk_length=1000, train_frac=0.8):
    filenames = get_videos(directory)

    chunks = []
    for fname in filenames:
        if fps(fname) >= framerate:
            print(fname)
            v = Video(fname, [3, image_width, image_width])
            for i in range(0, len(v), chunk_length):
                # chunk = VideoChunk(v[i : i + chunk_length], framerate)
                chunk = DiskVideoChunk(v[i : i + chunk_length], framerate,
                                       directory + '/_chunk_' + str(len(chunks)))
                                    #    directory + '/_overhead_chunk_' + str(len(chunks)))
                chunks.append(chunk)

    test_frac = 1 - train_frac
    test_chunk_indices = random.sample(list(range(len(chunks))),
                                       math.ceil(len(chunks) * test_frac))
    test_chunk_indices = set(test_chunk_indices)
    train_chunk_indices = set(list(range(len(chunks))))
    train_chunk_indices = train_chunk_indices - test_chunk_indices
    print("Training on indices: ")
    print(train_chunk_indices)
    print("Testing on indices: ")
    print(test_chunk_indices)

    train_chunks = [chunks[i] for i in train_chunk_indices]
    test_chunks = [chunks[i] for i in test_chunk_indices]

    train_dataset = ChunkData(train_chunks, seq_len, framerate, image_width)
    test_dataset = ChunkData(test_chunks, seq_len, framerate, image_width)
    return train_dataset, test_dataset

def load_disk_backed_data(checkpoint_path):
    data_checkpoint = torch.load(checkpoint_path)
    train_data = data_checkpoint['train_data']
    test_data = data_checkpoint['test_data']

    # this path will initially include the disk it's on
    # strip that folder name and use its current location instead
    dataset_root = os.path.dirname(checkpoint_path)
    for chunk in train_data.videos:
        chunk.path = dataset_root + os.path.basename(chunk.path)
    for chunk in test_data.videos:
        chunk.path = dataset_root + os.path.basename(chunk.path)
    return train_data, test_data


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

        # pull out the entire subsequence we're going to be taking chunks from
        # this prevents thrashing on the disk-backed DiskVideoChunk
        skip_rate = math.ceil(v.fps / self.framerate)
        video_subset = v[start_frame : start_frame + self. seq_len * skip_rate]

        seq = torch.Tensor(self.seq_len, *self.image_size)
        for t in range(self.seq_len):
            skipped_t = t * skip_rate
            frame = video_subset[skipped_t]
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



# data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/soccer'
# data_path = '/speedy/data/soccer'

# data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/basketball'
# train_data, test_data = make_split_datasets(data_path, 5, framerate=15, chunk_length=50)
#
# print("Number of training sequences (with overlap): " + str(len(train_data)))
# print("Number of testing sequences (with overlap): " + str(len(test_data)))
#
# save_dict = {
#         'train_data': train_data,
#         'test_data': test_data,
#     }
# torch.save(save_dict, data_path + '/dataset.t7')


# data_path = '/Users/willw/Downloads/dummy_data'
# # data_path = '/speedy/data/soccer'
#
# # data_path = '/misc/vlgscratch3/FergusGroup/wwhitney/basketball'
# train_data, test_data = make_split_datasets(data_path, 5, framerate=2, chunk_length=50)
#
# #
# # print("Number of training sequences (with overlap): " + str(len(train_data)))
# # print("Number of testing sequences (with overlap): " + str(len(test_data)))
# #
# save_dict = {
#         'train_data': train_data,
#         'test_data': test_data,
#     }
# torch.save(save_dict, data_path + '/dataset.t7')


# data_path = '/Users/willw/Downloads/APIDIS_VIDEO'
# train_data, test_data = make_split_datasets(data_path, 5, framerate=15, chunk_length=50)
#
# print("Number of training sequences (with overlap): " + str(len(train_data)))
# print("Number of testing sequences (with overlap): " + str(len(test_data)))
#
# save_dict = {
#         'train_data': train_data,
#         'test_data': test_data,
#     }
# torch.save(save_dict, data_path + '/overhead.t7')
