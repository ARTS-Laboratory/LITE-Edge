# Copyright (c) UofSC ARTS Lab, 2025
# Signal Compensation LSTM model training

import keras
import numpy as np
import scipy.signal as signal
from training_utils import WindowGenerator

input_data = np.load('./../dataset/benchtop_test_v6/Test 2/package.npy')
filter_b, filter_a = signal.butter(N=1, Wn=10, btype='highpass', fs=400)

input_filtered = signal.filtfilt(filter_b, filter_a, input_data).reshape(1, -1, 1)
output_data = np.load('./../dataset/benchtop_test_v6/Test 2/reference.npy').reshape(1, -1, 1)

windows = WindowGenerator(input_filtered, output_data, window_size=1200)

model = keras.Sequential([
    keras.layers.Input(shape=(None,1)),

    keras.layers.LSTM(units=50,
                      stateful = False,
                      return_sequences=True
                      ),

    keras.layers.Dense(units=1)
    ])

model.summary()
model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.mean_squared_error
        )

model.fit(
        windows,
        shuffle=True,
        epochs=100
        )

model.save('./model.keras')
