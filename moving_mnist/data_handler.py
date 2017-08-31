import sys
import numpy as np
import h5py
import sys
import datetime
from random import randint

import scipy.misc


class BouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, seq_length = 20, output_image_size=64):
        self.seq_length_ = seq_length
        self.image_size_ = 64
        self.output_image_size = output_image_size
        self.num_digits_ = 2
        self.step_length_ = 0.1

        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        f = h5py.File('./moving_mnist/mnist.h5')

        self.data_ = f['train'].value.reshape(-1, 28, 28)
        f.close()
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        np.random.shuffle(self.indices_)

    def GetDims(self):
        return self.frame_size_

    def GetSeqLength(self):
        return self.seq_length_

    def GetRandomTrajectory(self, num_digits):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(num_digits)
        x = np.random.rand(num_digits)

        # Choose a random velocity.
        theta = np.random.rand(num_digits) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, num_digits))
        start_x = np.zeros((length, num_digits))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            for j in range(num_digits):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)
        # return b

    def Resize(self, data, size):
        output_data = np.zeros((self.seq_length_, size, size), 
                               dtype=np.float32)

        for i, frame in enumerate(data):
            output_data[i] = scipy.misc.imresize(frame, (size, size))

        return output_data

    def GetItem(self, verbose=False):
        start_y, start_x = self.GetRandomTrajectory(self.num_digits_)

        # minibatch data
        data = np.zeros((self.seq_length_,
                         self.image_size_, self.image_size_), dtype=np.float32)

        for n in range(self.num_digits_):

            # get random digit from dataset
            ind = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                np.random.shuffle(self.indices_)
            digit_image = self.data_[ind, :, :]

            # generate video
            for i in range(self.seq_length_):
                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right] = self.Overlap(
                    data[i, top:bottom, left:right], digit_image)

        if self.output_image_size == self.image_size_:
            return data
        else:
            return self.Resize(data, self.output_image_size)

class ColoredBouncingMNISTDataHandler(object):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, seq_length = 20, output_image_size=64):
        self.seq_length_ = seq_length
        self.image_size_ = 64
        self.output_image_size = output_image_size
        self.num_digits_ = 2
        self.step_length_ = 0.1

        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        f = h5py.File('./moving_mnist/mnist.h5')

        self.data_ = f['train'].value.reshape(-1, 28, 28)
        f.close()
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        np.random.shuffle(self.indices_)

    def GetDims(self):
        return self.frame_size_

    def GetSeqLength(self):
        return self.seq_length_

    def GetRandomTrajectory(self, num_digits):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

        # Initial position uniform random inside the box.
        y = np.random.rand(num_digits)
        x = np.random.rand(num_digits)

        # Choose a random velocity.
        theta = np.random.rand(num_digits) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, num_digits))
        start_x = np.zeros((length, num_digits))
        for i in range(length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            for j in range(num_digits):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def Overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)
        # return b

    def Resize(self, data, size):
        output_data = np.zeros((self.seq_length_, size, size, 3), 
                               dtype=np.float32)

        for i, frame in enumerate(data):
            output_data[i] = scipy.misc.imresize(frame, (size, size))

        return output_data

    def GetItem(self, verbose=False):
        start_y, start_x = self.GetRandomTrajectory(self.num_digits_)

        # minibatch data
        data = np.zeros((self.seq_length_, 
                         self.image_size_, 
                         self.image_size_,
                         3),
                        dtype=np.float32)

        for n in range(self.num_digits_):

            # get random digit from dataset
            ind = self.indices_[self.row_]
            self.row_ += 1
            if self.row_ == self.data_.shape[0]:
                self.row_ = 0
                np.random.shuffle(self.indices_)
            digit_image = self.data_[ind, :, :]

            # generate video
            for i in range(self.seq_length_):
                top = start_y[i, n]
                left = start_x[i, n]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                data[i, top:bottom, left:right, n] = self.Overlap(
                    data[i, top:bottom, left:right, n], digit_image)

        if self.output_image_size == self.image_size_:
            return data
        else:
            return self.Resize(data, self.output_image_size)
