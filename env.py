import torch
from torch.utils.data import Dataset, DataLoader

import pdb
import math
import random
import numpy as np

from PIL import Image

class Thing:
    def __init__(self, color, loc, vel):
        self.color = color
        self.loc = loc
        self.vel = vel
        self.will_randomize = False
        self.regular_color = color
        self.randomize_color = [1, 0, 0]

def normalize_color(c):
    if type(c) == list:
        c = torch.Tensor(c)
    c = c / c.norm()
    return list(c)

def random_color():
    color = torch.Tensor(3).uniform_(0,1)
    return normalize_color(color)


# fixed_color = [random.uniform(0, 1) for _ in range(3)]
fixed_colors = [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 1]]
fixed_colors = [normalize_color(c) for c in fixed_colors]
# fixed_colors = [random_color() for _ in range(4)]

class DataGenerator:
    def __init__(self, balls, colors, image_width):
        self.size = image_width
        self.image_size = [3, self.size, self.size]
        self.n_things = balls
        self.max_speed = 1
        self.thing_size = 2

        self.colors = colors
        # 'vary'
        # self.colors = 'random'
        # self.colors = 'white'

    def make_thing(self, color=None):
        if color is None:
            color = random_color()
        return Thing(
            color = color,
            loc = [random.randint(0, self.size - self.thing_size),
                  random.randint(0, self.size - self.thing_size)],
            vel = [random.randint(-self.max_speed, self.max_speed),
                  random.randint(-self.max_speed, self.max_speed)]
        )

    def bounce(self, thing):
        for index in range(len(thing.loc)):
            if thing.loc[index] + (self.thing_size - 1) >= self.size:
                thing.vel[index] = - thing.vel[index]
                thing.loc[index] += 2 * thing.vel[index]
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
        canvas = torch.zeros(*self.image_size)
        for thing in self.things:
            thing_stamp = torch.Tensor(thing.color)
            thing_stamp = thing_stamp.unsqueeze(1).unsqueeze(1)
            thing_stamp = thing_stamp.expand(3, 
                                             self.thing_size, 
                                             self.thing_size)
            try:
                canvas[:, 
                       thing.loc[0] : thing.loc[0] + self.thing_size, 
                       thing.loc[1] : thing.loc[1] + self.thing_size, 
                      ] = thing_stamp
            except:
                pdb.set_trace()
        return canvas

class RandomizingGenerator(DataGenerator):
    def step(self):
        for i, thing in enumerate(self.things):
            if thing.will_randomize:
                thing.loc = self.make_thing(thing.color).loc
                thing.will_randomize = False
            else:
                if random.randint(1, 4) == 1:
                    thing.will_randomize = True
                thing.loc = [thing.loc[i] + thing.vel[i] for i in range(2)]
                self.bounce(thing)

            if self.colors == 'vary':
                self.increment_color(thing)
        return self

    def render(self):
        canvas = torch.zeros(*self.image_size)
        randomizing = False
        for thing in self.things:
            if thing.will_randomize:
                thing_stamp = torch.Tensor(thing.randomize_color)
                randomizing = True
            else:
                thing_stamp = torch.Tensor(thing.color)

            thing_stamp = thing_stamp.unsqueeze(1).unsqueeze(1)
            thing_stamp = thing_stamp.expand(3, 
                                             self.thing_size, 
                                             self.thing_size)
            try:
                canvas[:, 
                       thing.loc[0] : thing.loc[0] + self.thing_size, 
                       thing.loc[1] : thing.loc[1] + self.thing_size, 
                      ] = thing_stamp
            except:
                pdb.set_trace()
        return canvas, randomizing

class HorizontalLinesGenerator(DataGenerator):
    def __init__(self, balls, colors, image_width):
        super(HorizontalLinesGenerator, self).__init__(balls,
                                                       colors,
                                                       image_width)

    def start(self):
        if self.colors == 'vary':
            colors = [random.choice(fixed_colors) 
                      for i in range(self.n_things)]
        elif self.colors == 'random':
            colors = fixed_colors[:self.n_things]
        elif self.colors == 'white':
            colors = [[1,1,1] for _ in range(self.n_things)]

        self.things = []
        for i in range(self.n_things):
            thing = Thing(
                color=colors[i],
                loc=[i * math.floor(self.size / self.n_things),
                     random.randint(0, self.size - self.thing_size)],
                vel=[0,
                     random.randint(-self.max_speed, self.max_speed)])
            self.things.append(thing)
        return self

class BounceData(Dataset):
    def __init__(self, seq_len, balls, colors, image_width):
        self.seq_len = seq_len
        self.gen = DataGenerator(balls, colors, image_width)


    def __getitem__(self, i):
        canvas = torch.zeros(self.seq_len, *self.gen.image_size)
        canvas[0].copy_(self.gen.start().render())
        for t in range(1, self.seq_len):
            canvas[t].copy_(self.gen.step().render())
        return canvas


    def __len__(self):
        return 1000000

class HorizontalBounceData(BounceData):
    def __init__(self, seq_len, balls, colors, image_width):
        super(HorizontalBounceData, self).__init__(
            seq_len, balls, colors, image_width)
        self.gen = HorizontalLinesGenerator(balls, colors, image_width)

class RandomizeBounceData(BounceData):
    def __init__(self, seq_len, balls, colors, image_width):
        super(RandomizeBounceData, self).__init__(
            seq_len, balls, colors, image_width)
        self.gen = RandomizingGenerator(balls, colors, image_width)

    def __getitem__(self, i):
        canvas = torch.zeros(self.seq_len, *self.gen.image_size)
        self.gen.start()
        randomizing = []
        for t in range(self.seq_len):
            frame, randomized = self.gen.render()
            canvas[t].copy_(frame)
            randomizing.append(randomized)
            self.gen.step()
        return canvas, randomizing

# gen = DataGenerator()
# gen.start()
# show(gen.render())
# show(gen.step().render())

# gen = HorizontalLinesGenerator(4, 'random', 8)
# gen.start()
# show(gen.render())
# for i in range(20):
#     show(gen.step().render())
