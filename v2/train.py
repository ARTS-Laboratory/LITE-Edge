# Copyright (c) UofSC ARTS Lab, 2025
# Signal Compensation LSTM model training

import keras
import numpy as np
from training_utils import WindowGenerator

input_data = np.load('./../dataset/benchtop_test_v6/Test 2/package.npy').reshape(1, -1, 1)
output_data = np.load('./../dataset/benchtop_test_v6/Test 2/reference.npy').reshape(1, -1, 1)

windows = WindowGenerator(input_data, output_data, window_size=1200)

model = keras.Sequential([
    keras.layers.Input(shape=(None,1)),
    keras.layers.LSTM(units=50),
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
        epochs=3
        )

model.save('./model.keras')
