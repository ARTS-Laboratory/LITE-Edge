# Copyright (c) UofSC ARTS Lab, 2025 - 2026
# Signal Compensation LSTM model training

import keras
import numpy as np
import scipy.signal as signal
from training_utils import WindowGenerator, ax_filter

input_data = np.load('./../dataset/benchtop_test_v6/Test 2/package.npy')

input_filtered = ax_filter(input_data, 1)
output_data = np.load('./../dataset/benchtop_test_v6/Test 2/reference.npy').reshape(1, -1, 1)

windows = WindowGenerator(input_filtered, output_data, window_size=1200)

# These provide the N value of the butterworth filter. Conceptually, the lower
# this value is, the higher the quality of the virtual accelerometer. This is
# because the filtering effect in the lower frequencies is less dramatic.

qualities = [1, 2, 3, 4, 5]

for q in qualities:
        input_filtered = ax_filter(input_data, q)
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
        model.save('./ax' + str(q) + '.keras')
