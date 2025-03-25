from pickletools import TAKEN_FROM_ARGUMENT1
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import random
import reduction_utils
from svd_classes import ReducedLSTMCell, SingularLSTM
from reduction_utils import compute_gradients, EliminationRule, gen_bc
import os
from training_utils import TrainingGenerator

# tf.compat.v1.disable_eager_execution()
from svd_classes import make_LSTM_singular_model, make_LSTM_reduced_model
tf.config.run_functions_eagerly(True)
X_test = np.load("./dataset/V4/X_test.npy").reshape(1, -1, 1)
Y_test = np.load("./dataset/V4/Y_test.npy").reshape(1, -1, 1)

testing_batches = TrainingGenerator(X_test, Y_test, train_len=400)

test_batch_x, test_batch_y = testing_batches[random.randint(0, len(testing_batches) - 1)]

window_x = test_batch_x
window_y = test_batch_y

model = keras.models.load_model("./model_saves/model")

print(model.summary())
w, u, bias = model.layers[0].get_weights()

print(w.shape)
print(u.shape)
print(bias.shape)

merged_w = np.append(w, u, axis=0).T
print('merged_w:', merged_w.shape)

u, s, vt = np.linalg.svd(merged_w)

s = np.diag(s)

target_rank = np.linalg.matrix_rank(merged_w) - 25

reduced_s = s[:target_rank, :target_rank]
reduced_u = u[:, :target_rank]
reduced_vt = vt[:target_rank, :]

print((reduced_u @ reduced_s @ reduced_vt).shape)

b, c = gen_bc(reduced_u, reduced_s, reduced_vt, target_rank)

print('Memory footprint:', b.size + c.size, 'weights')

print(b.shape, c.shape)

cell = ReducedLSTMCell(50, w=[b.T, c.T], b=bias, kernel_type=2)
reduced_lstm = SingularLSTM(50, cell=cell, return_sequences=True)

reduced_model = keras.models.Sequential()
reduced_model.add(
    keras.layers.InputLayer(input_shape=[None, model.input_shape[-1]])
)

reduced_model.add(reduced_lstm)

dense_top = keras.layers.TimeDistributed(keras.layers.Dense(1))

reduced_model.add(dense_top)
dense_top.set_weights(
    [
        model.layers[-1].weights[0].numpy(),
        model.layers[-1].weights[1].numpy(),
    ]
)


data_out = 'reduction_results/direct'
os.makedirs(data_out, exist_ok=True)
np.savetxt(data_out + '/measured.csv', np.array(window_x).squeeze())
np.savetxt(data_out + '/true.csv', np.array(window_y).squeeze())
np.savetxt(data_out + '/uncompressed.csv', model.predict(window_x).squeeze())
np.savetxt(data_out + '/reduced.csv', reduced_model.predict(window_x).squeeze())
