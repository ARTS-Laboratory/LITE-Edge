# Copyright (c) UofSC ARTS Lab, 2025 - 2026
# Classes and tools used for training the model

import keras
import numpy as np
from math import floor, ceil
import scipy.signal as signal
from random import shuffle

# Splits the dataset into a series of windows of a given size
class WindowGenerator(keras.utils.Sequence):
    def __init__(self, x: np.ndarray, y: np.ndarray, window_size: int=1200, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.x = x
        self.y = y


    def __len__(self):
        return floor(self.x.size / self.window_size)


    def __getitem__(self, index):
        return (self.x[:,index * self.window_size: index * self.window_size + self.window_size,:],
                self.y[:,index * self.window_size: index * self.window_size + self.window_size,:])

# Apply butterworth filter to the package dataset.
def ax_filter(package_data: np.ndarray, N: int) -> np.ndarray:
    # For our purposes, we will hardcode the critical frequency to 10Hz, which
    # is in the middle of our frequency range of interest. We will also fix
    # the frequency to 400Hz.
    filter_b, filter_a = signal.butter(N, 10, btype='highpass', fs=400)

    return signal.filtfilt(filter_b, filter_a, package_data).reshape(1, -1, 1)
