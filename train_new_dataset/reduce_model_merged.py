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
from export_model import export_binary

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
full_prediction = model.predict(window_x).squeeze()

print(model.summary())
w, u, bias = model.layers[0].get_weights()

print(w.shape)
print(u.shape)
print(bias.shape)

export_binary(model)

merged_w = np.append(w, u, axis=0).T
print('merged_w:', merged_w.shape)
matrix_rank =  np.linalg.matrix_rank(merged_w)

error = np.zeros(matrix_rank, dtype=np.float32)

for i in range(1, matrix_rank):
    u, s, vt = np.linalg.svd(merged_w)

    s = np.diag(s)

    target_rank = matrix_rank - i

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

    reduced_prediction = reduced_model.predict(window_x).squeeze()
    error[i] = np.mean((full_prediction - reduced_prediction) ** 2)

    data_out = 'merged_reductions/rank_' + str(target_rank) + '/'
    os.makedirs(data_out, exist_ok=True)

    np.int8(target_rank).tofile(data_out + 'rank.dat')
    b.flatten().astype('<f4').tofile(data_out + 'b.dat')
    c.flatten().astype('<f4').tofile(data_out + 'c.dat')
    np.savetxt(data_out + 'reduced.csv', reduced_prediction)

data_out = 'merged_reductions'
np.savetxt(data_out + '/measured.csv', np.array(window_x).squeeze())
np.savetxt(data_out + '/true.csv', np.array(window_y).squeeze())
np.savetxt(data_out + '/uncompressed.csv', full_prediction)
np.savetxt('merged_reductions/error.csv', error)
