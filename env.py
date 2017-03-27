import torch
import math
import random
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

class Thing:
    def __init__(self, color, loc, vel):
        self.color = color
        self.loc = loc
        self.vel = vel

def random_color():
    color = torch.Tensor(3).uniform_(0,1)
    if sum(color) < 1:
        color = color / sum(color)
    return color

# fixed_color = [random.uniform(0, 1) for _ in range(3)]
fixed_color = [1.0 for _ in range(3)]
class DataGenerator:
    def __init__(self):
        self.size = 6
        self.n_things = 2
        self.max_speed = 1

    def make_thing(self):
        return Thing(
            color = fixed_color,
            # color = random_color(),
            loc = [random.randint(0, self.size - 1),
                  random.randint(0, self.size - 1)],
            vel = [random.randint(-self.max_speed, self.max_speed),
                  random.randint(-self.max_speed, self.max_speed)]
        )

    def bounce(self, thing):
        for index in range(len(thing.loc)):
            if thing.loc[index] >= self.size:
                thing.vel[index] = - thing.vel[index]
                thing.loc[index] = 2 * self.size - 2 - thing.loc[index]
            if thing.loc[index] < 0:
                thing.vel[index] = - thing.vel[index]
                thing.loc[index] = - thing.loc[index]

    def start(self):
        self.things = [self.make_thing() for i in range(self.n_things)]
        return self

    def step(self):
        # return self
        for thing in self.things:
            thing.loc = [thing.loc[i] + thing.vel[i] for i in range(2)]
            self.bounce(thing)
        return self

    def render(self):
        canvas = torch.zeros(3, self.size, self.size)
        for thing in self.things:
            canvas[:, thing.loc[0], thing.loc[1]] = torch.Tensor(thing.color)
        return canvas

def show(tensor):
    plt.figure()
    return plt.imshow(tensor.numpy())

# gen = DataGenerator()
# gen.start()
# show(gen.render())
# show(gen.step().render())
