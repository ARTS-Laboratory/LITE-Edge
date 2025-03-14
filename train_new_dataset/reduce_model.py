# Copyright UofSC ARTS Lab, 2024

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow as tf
import random
from reduction_utils import compute_gradients, EliminationRule

from training_utils import TrainingGenerator

# tf.compat.v1.disable_eager_execution()
from svd_classes import make_LSTM_singular_model, make_LSTM_reduced_model

"""
On merged kernel models.
"""


# %% load data
# use the formula SNR= (A_signal/A_noise)_rms^2. returned in dB
def signaltonoise(signal, noisy_signal, invert=False, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    if not invert:
        snr = (a_sig / a_noise) ** 2
    else:
        snr = (a_noise / a_sig) ** 2
    if not dB:
        return snr
    return 10 * math.log(snr, 10)


# Load training and validation data
X_train = np.load("./dataset/V4/X_train.npy")
Y_train = np.load("./dataset/V4/Y_train.npy")
X_test = np.load("./dataset/V4/X_test.npy").reshape(1, -1, 1)
Y_test = np.load("./dataset/V4/Y_test.npy").reshape(1, -1, 1)

fs = X_train.shape[1]

X_train = X_train[:,:fs//2].reshape(10, -1, 1)
Y_train = Y_train[:,:fs//2].reshape(10, -1, 1)

t_train = np.array([1/400*i for i in range(X_train.shape[1])])
t_test = np.array([1/400*i for i in range(X_test.shape[1])])

training_batches = TrainingGenerator(X_train, Y_train, train_len=400)
testing_batches = TrainingGenerator(X_test, Y_test, train_len=400)

train_batch_x, train_batch_y = training_batches[random.randint(0, len(training_batches) - 1)]
test_batch_x, test_batch_y = testing_batches[random.randint(0, len(testing_batches) - 1)]

model = keras.models.load_model("./model_saves/model")
print(model.summary())

full_model_train_prediction = model.predict(train_batch_x).flatten()
full_model_test_prediction = model.predict(test_batch_x)

np.save("./model_predictions/full model prediction.npy", full_model_train_prediction)
np.save("./model_predictions/full model windowed prediction.npy", full_model_test_prediction)
full_model_error = np.mean((train_batch_y - full_model_train_prediction) ** 2)
fec = np.mean((test_batch_y - full_model_test_prediction) ** 2)

# Relative to starting error,
e_thresh = 0.005  # allowed increase in error for one sigma
e_tot = 0.2  # allowable total error increase

power = 2  # power of Taylor approximation

smodel = make_LSTM_singular_model(model, kernel_type=2, return_sequences=True)

tensor_kernels = [layer.cell.kernel for layer in smodel.layers[:-1]]
kernels = [layer.cell.kernel.numpy() for layer in smodel.layers[:-1]]
elim_rule = EliminationRule(kernels)

# TODO: Generalize this
n1 = 51
m1 = 200

ranks = [min(n1, m1)]
full_ranks = ranks.copy()
# shape [layer, [left, right], [n, m]]
weight_shapes = np.array(
    [[[n1, ranks[0]], [ranks[0], m1]]]
)
n_full_weights = (
    weight_shapes[0, 0] * weight_shapes[0, 1]
    # + weight_shapes[1, 0, 0] * weight_shapes[1, 1, 1]
)
# ranks of each kernel (starts out full)


# loop values
end_condition = False
iteration = 1
max_iters = sum(ranks)
history = {
    "error": [full_model_error],
    "n_weights": [0],  # number of eliminated weights
    "heuristic": [],
    "essential": [],
}


while not end_condition:
    # Select new batch to prevent overfitting on a particular data window
    Xc, yc = testing_batches[random.randint(0, len(testing_batches) - 1)]
    print("computing gradients...")
    grads = compute_gradients(Xc, yc, tensor_kernels, smodel, power=power)
    print("finished computing gradients...")

    # convert grads into numpy
    error = grads[0].numpy()
    grads = grads[1:]
    for grad in grads:
        for i, g in enumerate(grad):
            grad[i] = g.numpy()

    # calculate heuristic
    h_index, h = elim_rule.heuristic(grads, smodel)

    # eliminate proposed sigma
    elim_sigma = kernels[h_index[0]][0, h_index[1]]
    kernels[h_index[0]][0, h_index[1]] = 0
    tensor_kernels[h_index[0]].assign(kernels[h_index[0]])
    elim_rule.kernel_mask[h_index[0]][h_index[1]] = False
    ranks[h_index[0]] -= 1

    # make new prediction
    print("calculating prediction with sigma removed...")
    new_y_pred = smodel.predict(Xc)
    new_error = keras.losses.MeanSquaredError()(yc, new_y_pred).numpy()

    error_dif = (new_error - error) / fec
    essential = error_dif > e_thresh
    # reinstatiate sigma if increase in error was too high
    if essential:
        kernels[h_index[0]][0, h_index[1]] = elim_sigma
        tensor_kernels[h_index[0]].assign(kernels[h_index[0]])
        elim_rule.kernel_mask[h_index[0]][h_index[1]] = True
        elim_rule.essential_sigmas[h_index[0]][h_index[1]] = True
        ranks[h_index[0]] += 1

    # update history
    history["heuristic"].append(h)
    history["essential"].append(essential)
    history["error"].append(new_error)
    history["n_weights"].append(
        (n1 - ranks[0]) * (m1 - ranks[0])
    )

    # evaluate end condition
    end_condition = new_error / fec > (1 + e_tot) and not essential
    end_condition = end_condition or iteration >= max_iters

    if end_condition:
        print("error exceeds threshold, terminating pruning.")

    # print info to screen
    print("iteration %d finished" % iteration)
    print("layer of eliminated sigma:", h_index[0])
    print("index of eliminated sigma:", h_index[1])
    print("heuristic value:", h)
    print("error after elimination:", new_error)
    print("error difference:", error_dif)
    print("total error increase (percent):", (new_error / fec - 1) * 100)
    print("total weights eliminated:", history["n_weights"][-1])
    if essential:
        print("sigma was found to be essential and was not eliminated.")
    else:
        print("sigma was not found to be essential.")
    print()
    iteration += 1
# %%
# save stuff
error_history = np.array(history["error"])
n_weights_history = np.array(history["n_weights"])
heuristic_history = np.array(history["heuristic"])
essential_history = np.array(history["essential"])
np.save("./svd reduction/error.npy", error_history)
np.save("./svd reduction/n_weights.npy", n_weights_history)
np.save("./svd reduction/heuristic.npy", heuristic_history)
np.save("./svd reduction/is_essential.npy", essential_history)

# make some plots

X = test_batch_x
y = test_batch_y
t = t_test[0:400]

sy = smodel.predict(test_batch_x).flatten()
se = np.mean((y - sy) ** 2)

print("singular error on test:", se)
print("increase in error (percent):",
      (se - full_model_error) / full_model_error * 100)

full_model_test_prediction = model.predict(test_batch_x)
full_error = (y - full_model_test_prediction) ** 2
reduced_error = (y - sy) ** 2

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# prediction of full and reduced model
plt.figure(figsize=(10, 6))
plt.plot(t, y.squeeze(), label="true", linewidth=0.5)
plt.plot(t, full_model_test_prediction.squeeze(), label="uncompressed")
plt.plot(t, sy.squeeze(), label="reduced")
plt.legend()
plt.tight_layout()
plt.xlabel("time (s)")
plt.ylabel(r'acceleration ($m/s^2$)')
plt.savefig("./plots/prediction.png", dpi=300)

# Error comparison between full and reduced model
plt.figure(figsize=(10, 6))
plt.plot(t, full_error.squeeze(), label="uncompressed", color='orange')
plt.plot(t, reduced_error.squeeze(), label="reduced", color='green')
plt.legend()
plt.tight_layout()
plt.xlabel("time (s)")
plt.ylabel(r'error')
plt.savefig("./plots/error.png", dpi=300)

# error history

error_increase = (error_history - fec) / fec * 100
plt.figure(figsize=(4, 3))
plt.plot(
    n_weights_history[1:][np.logical_not(essential_history)],
    error_increase[1:][np.logical_not(essential_history)],
)
plt.tight_layout()
plt.xlabel("weights eliminated")
plt.ylabel("error increase")
plt.savefig("./plots/elimination-history.png", dpi=300)
# make the reduced model
keep_sigmas = elim_rule.kernel_mask
rmodel = make_LSTM_reduced_model(smodel, keep_sigmas, kernel_type=2)

ry = rmodel.predict(X).flatten()
re = np.mean((y - ry) ** 2)
# if you plot this you will see that its the same as sy.

n_full_weights = 0
for layer in model.layers:
    for weight in layer.weights:
        n_full_weights += weight.numpy().size

n_reduced_weights = 0
for layer in rmodel.layers:
    for weight in layer.weights:
        n_reduced_weights += weight.numpy().size

# see that this agrees with what was calculated during singular reduction
print("weights in full model:", n_full_weights)
print("weights in reduced model:", n_reduced_weights)
print(
    "percent reduction in weights: ",
    (n_full_weights - n_reduced_weights) / n_full_weights * 100,
)
