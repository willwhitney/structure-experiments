import torch
import math
import random
import numpy as np

from PIL import Image


from params import *

class Thing:
    def __init__(self, color, loc, vel):
        self.color = color
        self.loc = loc
        self.vel = vel

def random_color():
    color = torch.Tensor(3).uniform_(0,1)
    if sum(color) < 1:
        color = color / sum(color)
    return list(color)

# fixed_color = [random.uniform(0, 1) for _ in range(3)]
fixed_colors = [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]]
# fixed_colors = [random_color() for _ in range(4)]

class DataGenerator:
    def __init__(self):
        self.size = 6
        self.image_size = [3, self.size, self.size]
        self.n_things = opt.balls
        self.max_speed = 1

        self.colors = opt.colors
        # 'vary'
        # self.colors = 'random'
        # self.colors = 'white'

    def make_thing(self, color=None):
        if color is None:
            color = random_color()
        return Thing(
            color = color,
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

    def increment_color(self, thing):
        color_index = fixed_colors.index(thing.color)
        new_color_index = (color_index + 1) % len(fixed_colors)
        thing.color = fixed_colors[new_color_index]

    def start(self):
        # random colored balls
        if self.colors == 'vary':
            self.things = [self.make_thing(random.choice(fixed_colors))
                           for i in range(self.n_things)]

        # different colored balls
        if self.colors == 'random':
            self.things = [self.make_thing(fixed_colors[i])
                           for i in range(self.n_things)]

        # all white balls
        if self.colors == 'white':
            self.things = [self.make_thing([1,1,1])
                           for i in range(self.n_things)]
        return self

    def step(self):
        # return self
        for thing in self.things:
            thing.loc = [thing.loc[i] + thing.vel[i] for i in range(2)]
            self.bounce(thing)

            if self.colors == 'vary':
                self.increment_color(thing)
        return self

    def render(self):
        canvas = torch.zeros(3, self.size, self.size)
        for thing in self.things:
            canvas[:, thing.loc[0], thing.loc[1]] = torch.Tensor(thing.color)
        return canvas


# gen = DataGenerator()
# gen.start()
# show(gen.render())
# show(gen.step().render())
