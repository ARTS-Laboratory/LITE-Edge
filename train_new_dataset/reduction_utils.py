import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

"""
compute derivatives of the error up to power,
recursive
using tape/eager execution
"""
def compute_gradients(X, y_true, weights, model, power=3):
    if power == 1:  # base case
        with tf.GradientTape() as tape:
            tape.watch(weights)
            y_pred = model(X)
            error = keras.losses.MeanSquaredError()(y_true, y_pred)
        return [error, tape.gradient(error, weights)]
    else:
        with tf.GradientTape() as tape:
            tape.watch(weights)
            grads = compute_gradients(
                X, y_true, weights, model, power=power - 1
            )  # recursive step
        grads.append(tape.gradient(grads, weights))
        return grads


class EliminationRule:
    """
    pass kernels as list of numpy arrays
    """

    def __init__(self, kernels):
        self.kernels = kernels
        self.kernel_sizes = [k.shape[1] for k in kernels]
        self.kernel_mask = [
            np.ones(size, dtype=bool) for size in self.kernel_sizes
        ]

        self.essential_sigmas = [
            np.zeros(size, dtype=bool) for size in self.kernel_sizes
        ]

        self.tcc = [1, 1 / 2, 1 / 6]  # coefficients of Taylor approximation

    """
    pass grads as list of numpy arrays
    returns tuple index of suggested eliminated sigma
    """

    def heuristic(self, grads, smodel):
        # heuristic way to determine eliminated rank
        heuristic = [
            np.zeros(layer.cell.kernel.shape) for layer in smodel.layers[:-1]
        ]

        for p, grad in enumerate(grads):  # power
            for i, dkernel in enumerate(grad):  # layer
                heuristic[i] += self.tcc[p] * dkernel * self.kernels[i] ** (p + 1)

        hlayer_eliminate = 0
        hindex_eliminate = 0
        h_eliminate = None
        for i, hlayer in enumerate(heuristic):
            for j, h in enumerate(hlayer.flatten()):
                if (
                    (h_eliminate is None or h < h_eliminate)
                    and self.kernel_mask[i][j]
                    and not self.essential_sigmas[i][j]
                ):
                    hlayer_eliminate = i
                    hindex_eliminate = j
                    h_eliminate = h

        return (hlayer_eliminate, hindex_eliminate), h_eliminate
