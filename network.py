import random
import numpy as np

class network(object):
    def __init__(self, sizes, weight_initialization="random"):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weight_initialization == "random":
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        elif weight_initialization == "improved":
            self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

        else:
            raise ValueError("weight_initialization must be 'standard' or 'improved'")