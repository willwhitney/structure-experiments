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

# fixed_color = [random.uniform(0, 1) for _ in range(3)]
class DataGenerator:
    def __init__(self):
        self.size = 4
        self.n_things = 1
        self.max_speed = 0

    def make_thing(self):
        return Thing(
            # color = fixed_color,
            color = [random.uniform(0, 1) for _ in range(3)],
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
        for thing in self.things:
            thing.loc = [thing.loc[i] + thing.vel[i] for i in range(2)]
            self.bounce(thing)
        return self

    def render(self):
        canvas = torch.zeros(self.size, self.size, 3)
        for thing in self.things:
            canvas[thing.loc[0], thing.loc[1], :] = torch.Tensor(thing.color)
        return canvas

def show(tensor):
    plt.figure()
    return plt.imshow(tensor.numpy())

gen = DataGenerator()
gen.start()
show(gen.render())
show(gen.step().render())
