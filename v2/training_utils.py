# Copyright (c) UofSC ARTS Lab, 2025
# Classes and tools used for training the model

import keras
import numpy as np
from math import floor, ceil
from random import shuffle

# Splits the dataset into a series of windows of a given size
class WindowGenerator(keras.utils.PyDataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int=1200):
        self.window_size = window_size
        self.x = x
        self.y = y


    def __len__(self):
        return ceil(self.x.size / self.window_size)


    def __getitem__(self, index):
        return (self.x[index * self.window_size: index * self.window_size + self.window_size],
                self.y[index * self.window_size: index * self.window_size + self.window_size])
