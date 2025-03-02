# Copyright UofSC ARTS Lab, 2024

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTMCell
import tensorflow.keras.backend as backend

# from keras.layers.rnn import rnn_utils
# from keras.layers.rnn import gru_lstm_utils
import numpy as np

"""
v4: merged, half-merged, and split kernels
"""
"""
kernel_type: 0 split kernel
             1 half-merged
             2 merged
weights: for split kernel and half-merged kernel:
    (kernel, recurrent_kernel, w_left, w_right, u_left, u_right, bias)
for merged kernel:
    (kernel, w_left, w_right, bias)

"""


class SingularLSTMCell(LSTMCell):

    def __init__(
        self,
        units,
        weights,
        kernel_type=2,
        train_uv=False,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        uv_regularizer=None,
        **kwargs
    ):
        super(SingularLSTMCell, self).__init__(units, **kwargs)
        self.weights_ = weights
        self.train_uv = train_uv
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.kernel_type = kernel_type
        self.uv_regularizer = uv_regularizer

    # Maybe make another option for training b
    def build(self, input_shape):
        # default_caching_device = rnn_utils.caching_device(self)
        input_dim = input_shape[-1]
        if self.kernel_type == 0:  # split kernel
            self.kernel = self.add_weight(
                shape=(1, input_dim * 4),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                # caching_device=default_caching_device
            )
            self.recurrent_kernel = self.add_weight(
                shape=(1, self.units * 4),
                name="recurrent_kernel",
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint,
                # caching_device=default_caching_device
            )
            self.w_left = self.add_weight(
                shape=(input_dim, input_dim * 4),
                name="w_left",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.w_right = self.add_weight(
                shape=(
                    input_dim,
                    self.units * 4,
                ),  # same shape in split and merged
                name="w_right",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.u_left = self.add_weight(
                shape=(self.units, self.units * 4),
                name="u_left",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.u_right = self.add_weight(
                shape=(self.units, self.units * 4),
                name="u_right",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                trainable=self.train_uv,
            )
            self.set_weights(self.weights_)
        elif self.kernel_type == 1:  # half merged kernel
            self.kernel = self.add_weight(
                shape=(1, input_dim),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                # caching_device=default_caching_device
            )
            self.recurrent_kernel = self.add_weight(
                shape=(1, self.units),
                name="recurrent_kernel",
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint,
                # caching_device=default_caching_device
            )
            self.w_left = self.add_weight(
                shape=(input_dim, input_dim),
                name="w_left",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.w_right = self.add_weight(
                shape=(
                    input_dim,
                    self.units * 4,
                ),  # same shape in split and merged
                name="w_right",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.u_left = self.add_weight(
                shape=(self.units, self.units),
                name="u_left",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.u_right = self.add_weight(
                shape=(self.units, self.units * 4),
                name="u_right",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                trainable=self.train_uv,
            )
            self.set_weights(self.weights_)
        else:  # merged kernel
            self.kernel = self.add_weight(
                shape=(1, input_dim + self.units),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                # caching_device=default_caching_device
            )
            self.w_left = self.add_weight(
                shape=(
                    input_dim + self.units,
                    input_dim + self.units,
                ),
                name="w_left",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.w_right = self.add_weight(
                shape=(input_dim + self.units, self.units * 4),
                name="w_right",
                regularizer=self.uv_regularizer,
                trainable=self.train_uv,
            )
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name="bias",
                trainable=self.train_uv,
            )
            self.set_weights(self.weights_)

    def call(self, inputs, states, training=None):
        if self.kernel_type == 0:  # split kernel
            h_tm1 = states[0]  # previous memory state
            c_tm1 = states[1]  # previous carry state

            dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
            rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=4
            )

            if 0 < self.dropout < 1.0:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            wr_i, wr_f, wr_c, wr_o = tf.split(
                self.w_right, num_or_size_splits=4, axis=1
            )
            ws_i, ws_f, ws_c, ws_o = tf.split(
                self.kernel, num_or_size_splits=4, axis=1
            )
            wl_i, wl_f, wl_c, wl_o = tf.split(
                self.w_left, num_or_size_splits=4, axis=1
            )

            # keras uses left multiplication of horizontal (row) vectors
            x_i = backend.dot(inputs_i, wl_i)
            x_f = backend.dot(inputs_f, wl_f)
            x_c = backend.dot(inputs_c, wl_c)
            x_o = backend.dot(inputs_o, wl_o)

            x_i = x_i * ws_i
            x_f = x_f * ws_f
            x_c = x_c * ws_c
            x_o = x_o * ws_o

            x_i = backend.dot(x_i, wr_i)
            x_f = backend.dot(x_f, wr_f)
            x_c = backend.dot(x_c, wr_c)
            x_o = backend.dot(x_o, wr_o)

            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(
                    self.bias, num_or_size_splits=4, axis=0
                )
                x_i = backend.bias_add(x_i, b_i)
                x_f = backend.bias_add(x_f, b_f)
                x_c = backend.bias_add(x_c, b_c)
                x_o = backend.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.0:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1

            ur_i, ur_f, ur_c, ur_o = tf.split(
                self.u_right, num_or_size_splits=4, axis=1
            )
            us_i, us_f, us_c, us_o = tf.split(
                self.recurrent_kernel, num_or_size_splits=4, axis=1
            )
            ul_i, ul_f, ul_c, ul_o = tf.split(
                self.u_left, num_or_size_splits=4, axis=1
            )

            r_i = backend.dot(h_tm1_i, ul_i)
            r_f = backend.dot(h_tm1_f, ul_f)
            r_c = backend.dot(h_tm1_c, ul_c)
            r_o = backend.dot(h_tm1_o, ul_o)

            r_i = r_i * us_i
            r_f = r_f * us_f
            r_c = r_c * us_c
            r_o = r_o * us_o

            r_i = backend.dot(r_i, ur_i)
            r_f = backend.dot(r_f, ur_f)
            r_c = backend.dot(r_c, ur_c)
            r_o = backend.dot(r_o, ur_o)

            i = self.recurrent_activation(x_i + r_i)
            f = self.recurrent_activation(x_f + r_f)
            c = f * c_tm1 + i * self.activation(x_c + r_c)
            o = self.recurrent_activation(x_o + r_o)

            h = o * self.activation(c)

            return h, [h, c]
        elif self.kernel_type == 1:  # half-merged kernel
            h_tm1 = states[0]  # previous memory state
            c_tm1 = states[1]  # previous carry state

            dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
            rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=4
            )

            if 0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]

            # keras uses left multiplication of horizontal (row) vectors
            x = backend.dot(inputs, self.w_left)
            x = x * self.kernel
            x = backend.dot(x, self.w_right)

            if self.use_bias:
                x = backend.bias_add(x, self.bias)

            if 0 < self.recurrent_dropout < 1.0:
                h_tm1 = h_tm1 * rec_dp_mask[0]

            z = backend.dot(h_tm1, self.u_left)
            z = z * self.recurrent_kernel
            z = backend.dot(z, self.u_right)
            z += x

            z = tf.split(z, num_or_size_splits=4, axis=1)

            c, o = self._compute_carry_and_output_fused(z, c_tm1)

            h = o * self.activation(c)

            return h, [h, c]
        elif self.kernel_type == 2:  # merged kernel
            h_tm1 = states[0]  # previous memory state
            c_tm1 = states[1]  # previous carry state

            dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
            rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=4
            )

            if 0 < self.dropout < 1.0:
                inputs = inputs * dp_mask[0]
            if 0 < self.recurrent_dropout < 1.0:
                h_tm1 = h_tm1 * rec_dp_mask[0]

            x = tf.concat([inputs, h_tm1], axis=1)

            x = backend.dot(x, self.w_left)
            x = x * self.kernel
            x = backend.dot(x, self.w_right)

            if self.use_bias:
                x = backend.bias_add(x, self.bias)

            z = tf.split(x, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

            h = o * self.activation(c)

            return h, [h, c]


"""
Reduced LSTM Cell.
for merged kernel, keep u=None
"""


class ReducedLSTMCell(LSTMCell):

    def __init__(self, units, w=None, u=None, b=None, kernel_type=2, **kwargs):
        super(ReducedLSTMCell, self).__init__(units, **kwargs)
        self.w = w
        self.u = u
        self.b = b
        # self.weights = weights
        self.kernel_type = kernel_type

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.kernel_type == 0:  # split kernel
            self.w_left = []
            self.w_right = []
            self.u_left = []
            self.u_right = []
            weights = []
            for i in range(4):
                gate = ["i", "f", "c", "o"][i]
                rank_w = self.w[i][0].shape[1]
                rank_u = self.u[i][0].shape[1]
                self.w_left.append(
                    self.add_weight(
                        shape=(input_dim, rank_w),
                        name="w_left_" + gate,
                        trainable=False,
                    )
                )
                self.w_right.append(
                    self.add_weight(
                        shape=(rank_w, self.units - rank_w),
                        name="w_right_" + gate,
                        trainable=False,
                    )
                )
                self.u_left.append(
                    self.add_weight(
                        shape=(self.units, rank_u),
                        name="u_left_" + gate,
                        trainable=False,
                    )
                )
                self.u_right.append(
                    self.add_weight(
                        shape=(rank_u, self.units - rank_u),
                        name="u_right_" + gate,
                        trainable=False,
                    )
                )
                weights += [
                    self.w[i][0],
                    self.w[i][1],
                    self.u[i][0],
                    self.u[i][1],
                ]
            self.bias = self.add_weight(
                shape=(self.units * 4,), name="bias", trainable=False
            )
            weights.append(self.b)
            self.set_weights(weights)
        elif self.kernel_type == 1:  # half merged kernel
            rank_w = self.w[0].shape[1]
            rank_u = self.u[0].shape[1]

            self.w_left = self.add_weight(
                shape=(input_dim, rank_w),
                name="w_left",
                trainable=False,
            )
            self.w_right = self.add_weight(
                shape=(rank_w, self.units * 4 - rank_w),
                name="w_right",
                trainable=False,
            )
            self.u_left = self.add_weight(
                shape=(self.units, rank_u),
                name="u_left",
                trainable=False,
            )
            self.u_right = self.add_weight(
                shape=(rank_u, self.units * 4 - rank_u),
                name="u_right",
                trainable=False,
            )
            self.bias = self.add_weight(
                shape=(self.units * 4,), name="bias", trainable=False
            )
            weights = (
                self.w[0],
                self.w[1],
                self.u[0],
                self.u[1],
                self.b,
            )
            self.set_weights(weights)
        else:  # merged kernel
            rank = self.w[0].shape[1]

            self.w_left = self.add_weight(
                shape=(input_dim + self.units, rank),
                name="w_left",
                trainable=False,
            )
            self.w_right = self.add_weight(
                shape=(rank, self.units * 4 - rank),
                name="w_right",
                trainable=False,
            )
            self.bias = self.add_weight(
                shape=(self.units * 4,), name="bias", trainable=False
            )
            weights = (self.w[0], self.w[1], self.b)
            self.set_weights(weights)

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        if self.kernel_type == 0:  # split kernel
            x_i = backend.dot(inputs, self.w_left[0])
            x_f = backend.dot(inputs, self.w_left[1])
            x_c = backend.dot(inputs, self.w_left[2])
            x_o = backend.dot(inputs, self.w_left[3])

            x_i = tf.concat((x_i, backend.dot(x_i, self.w_right[0])), axis=1)
            x_f = tf.concat((x_f, backend.dot(x_f, self.w_right[1])), axis=1)
            x_c = tf.concat((x_c, backend.dot(x_c, self.w_right[2])), axis=1)
            x_o = tf.concat((x_o, backend.dot(x_o, self.w_right[3])), axis=1)

            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(
                    self.bias, num_or_size_splits=4, axis=0
                )
                x_i = backend.bias_add(x_i, b_i)
                x_f = backend.bias_add(x_f, b_f)
                x_c = backend.bias_add(x_c, b_c)
                x_o = backend.bias_add(x_o, b_o)

            z_i = backend.dot(h_tm1, self.u_left[0])
            z_f = backend.dot(h_tm1, self.u_left[1])
            z_c = backend.dot(h_tm1, self.u_left[2])
            z_o = backend.dot(h_tm1, self.u_left[3])

            z_i = tf.concat((z_i, backend.dot(z_i, self.u_right[0])), axis=1)
            z_f = tf.concat((z_f, backend.dot(z_f, self.u_right[1])), axis=1)
            z_c = tf.concat((z_c, backend.dot(z_c, self.u_right[2])), axis=1)
            z_o = tf.concat((z_o, backend.dot(z_o, self.u_right[3])), axis=1)

            z_i += x_i
            z_f += x_f
            z_c += x_c
            z_o += x_o

            z = (z_i, z_f, z_c, z_o)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)
            h = o * self.activation(c)

            return h, [h, c]
        elif self.kernel_type == 1:  # half merged kernel
            x = backend.dot(inputs, self.w_left)
            x = tf.concat((x, backend.dot(x, self.w_right)), axis=1)
            if self.use_bias:
                x = backend.bias_add(x, self.bias)
            z = backend.dot(h_tm1, self.u_left)
            z = tf.concat((z, backend.dot(z, self.u_right)), axis=1)
            z += x
            z = tf.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)
            h = o * self.activation(c)

            return h, [h, c]
        else:  # merged kernel
            x = tf.concat([inputs, h_tm1], axis=1)
            x = backend.dot(x, self.w_left)
            x = tf.concat((x, backend.dot(x, self.w_right)), axis=1)
            if self.use_bias:
                x = backend.bias_add(x, self.bias)
            z = x
            z = tf.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)
            h = o * self.activation(c)

            return h, [h, c]


"""
Modified from keras LSTM class for used with SingularLSTMCell and
ReducedLSTMCells.
"""


class SingularLSTM(keras.layers.LSTM):

    def __init__(self, units, cell=None, **kwargs):
        super(SingularLSTM, self).__init__(units, **kwargs)
        # self.kernel_regularizer = kwargs['kernel_regularizer']
        # self.recurrent_regularizer = kwargs['recurrent_regularizer']
        self.cell = cell
        # self.kernel = [cell.w_sigma, cell.u_sigma]

    """ copied from the non-gpu portion of keras.layers.LSTM call() function"""

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # The input should be dense, padded with zeros. If a ragged input is
        # fed into the layer, it is padded and the row lengths are used for
        # masking. inputs, row_lengths = backend.
        # convert_inputs_if_ragged(inputs) is_ragged_input =
        # (row_lengths is not None)
        # self._validate_args_if_ragged(is_ragged_input, mask)

        # LSTM does not support constants. Ignore it during process.
        inputs, initial_state, _ = self._process_inputs(
            inputs, initial_state, None
        )

        if isinstance(mask, list):
            mask = mask[0]

        input_shape = backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]

        # Fall back to use the normal LSTM.
        kwargs = {"training": training}
        self._maybe_reset_cell_dropout_mask(self.cell)

        def step(inputs, states):
            return self.cell(inputs, states, **kwargs)

        last_output, outputs, states = backend.rnn(
            step,
            inputs,
            initial_state,
            constants=None,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask,
        )
        # runtime = gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_UNKNOWN)
        if self.stateful:
            updates = [
                tf.compat.v1.assign(
                    self_state, tf.cast(state, self_state.dtype)
                )
                for self_state, state in zip(self.states, states)
            ]
            self.add_update(updates)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            return [output] + list(states)
        # elif self.return_runtime: # just gonna comment this out
        #   return output, runtime
        return output

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]


class PrunableTimeDistributed(keras.layers.TimeDistributed):

    def __init__(self, layer, **kwargs):
        super(PrunableTimeDistributed, self).__init__(layer, **kwargs)
        self.layer = layer

    def get_prunable_weights(self):
        return self.layer.layer.weights


"""
The Hoyer regularizer is the ratio of the L1 and L2 norms. It has the effect
of sparsifying the input tensor but does not reduce the tensor's energy.
"""


class HoyerRegularizer:
    def __init__(self, hoyer=None):
        hoyer = 0 if hoyer is None else hoyer
        self.hoyer = backend.cast_to_floatx(hoyer)

    def __call__(self, x):
        regularization = (
            self.hoyer * tf.reduce_sum(tf.abs(x)) / tf.reduce_sum(tf.square(x))
        )
        return regularization

    def get_config(self):
        return {"hoyer": self.hoyer}


"""
Helper method for make_LSTM_singular_model with kernel_type = 0.

this code is out of date and likely doesn't work but i don't want to fix it.
"""


def make_split_LSTM_singular_model(
    model, hoyer=None, orthogonal=None, return_sequences=False
):
    smodel = keras.models.Sequential()
    smodel.add(
        keras.layers.InputLayer(input_shape=[None, model.input_shape[-1]])
    )
    for i, layer in enumerate(model.layers[:-1]):
        w, u, b = layer.get_weights()

        # w = np.expand_dims(w, -1)
        # u = np.expand_dims(u, -1)

        # units = u.shape[0]
        units = layer.units

        w_split = [
            w[:, :units],
            w[:, units: units * 2],
            w[:, units * 2: units * 3],
            w[:, units * 3:],
        ]
        u_split = [
            u[:, :units],
            u[:, units: units * 2],
            u[:, units * 2: units * 3],
            u[:, units * 3:],
        ]

        wu = []
        for split in [w_split, u_split]:
            lefts = []
            sigmas = []
            rights = []
            for mat in split:
                left, sigma, right = np.linalg.svd(
                    mat, full_matrices=False, compute_uv=True
                )
                lefts.append(left)
                sigmas.append(np.expand_dims(sigma, axis=0))
                rights.append(right)
            unsplit_left = lefts[0]
            unsplit_sigma = sigmas[0]
            unsplit_right = rights[0]
            for left, sigma, right in zip(lefts[1:], sigmas[1:], rights[1:]):
                unsplit_left = np.append(unsplit_left, left, axis=1)
                unsplit_sigma = np.append(unsplit_sigma, sigma, axis=1)
                unsplit_right = np.append(unsplit_right, right, axis=1)
            wu.append([unsplit_left, unsplit_sigma, unsplit_right])

        # b = np.expand_dims(b, axis=-1).T
        # wu[0] = np.expand_dims(wu[0], 0)
        # wu[1] = np.expand_dims(wu[1], 0)
        if hoyer is not None and hoyer != 0:
            kernel_regularizer = HoyerRegularizer(hoyer)
            recurrent_regularizer = HoyerRegularizer(hoyer)
        else:
            kernel_regularizer = None
            recurrent_regularizer = None
        if orthogonal is not None and orthogonal != 0:
            uv_regularizer = keras.regularizers.OrthogonalRegularizer(
                factor=orthogonal, mode="rows"
            )
            train_uv = True
        else:
            uv_regularizer = None
            train_uv = False
        weights = (
            wu[0][1],
            wu[1][1],
            wu[0][0],
            wu[0][2],
            wu[1][0],
            wu[1][2],
            b,
        )
        cell = SingularLSTMCell(
            units,
            weights,
            kernel_type=0,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            train_uv=train_uv,
            uv_regularizer=uv_regularizer,
        )
        rs = True
        if i == len(model.layers[:-1]) - 1 and not return_sequences:
            rs = False
        lstm = SingularLSTM(units, cell=cell, return_sequences=rs)
        smodel.add(lstm)

    dense_top = keras.layers.Dense(1)
    if return_sequences:
        dense_top = keras.layers.TimeDistributed(dense_top)
    smodel.add(dense_top)
    dense_top.set_weights(
        [
            model.layers[-1].weights[0].numpy(),
            model.layers[-1].weights[1].numpy(),
        ]
    )
    return smodel


def make_half_merged_LSTM_singular_model(
    model, hoyer=None, orthogonal=None, return_sequences=False
):
    # %%
    smodel = keras.models.Sequential()
    smodel.add(
        keras.layers.InputLayer(input_shape=[None, model.input_shape[-1]])
    )
    for i, layer in enumerate(model.layers[:-1]):
        w, u, b = layer.get_weights()
        units = layer.units

        wu = []
        for mat in [w, u]:
            left, sigma, right = np.linalg.svd(
                mat, full_matrices=False, compute_uv=True
            )
            sigma = np.expand_dims(sigma, axis=0)
            wu.append([left, sigma, right])
        weights = (
            wu[0][1],
            wu[1][1],
            wu[0][0],
            wu[0][2],
            wu[1][0],
            wu[1][2],
            b,
        )
        if hoyer is not None and hoyer != 0:
            kernel_regularizer = HoyerRegularizer(hoyer)
            recurrent_regularizer = HoyerRegularizer(hoyer)
        else:
            kernel_regularizer = None
            recurrent_regularizer = None
        if orthogonal is not None and orthogonal != 0:
            uv_regularizer = keras.regularizers.OrthogonalRegularizer(
                factor=orthogonal, mode="rows"
            )
            train_uv = True
        else:
            uv_regularizer = None
            train_uv = False
        cell = SingularLSTMCell(
            units,
            weights,
            kernel_type=1,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            train_uv=train_uv,
            uv_regularizer=uv_regularizer,
        )
        rs = True
        if i == len(model.layers[:-1]) - 1 and not return_sequences:
            rs = False
        lstm = SingularLSTM(units, cell=cell, return_sequences=rs)
        smodel.add(lstm)

    dense_top = keras.layers.Dense(1)
    if return_sequences:
        dense_top = keras.layers.TimeDistributed(dense_top)
    smodel.add(dense_top)
    dense_top.set_weights(
        [
            model.layers[-1].weights[0].numpy(),
            model.layers[-1].weights[1].numpy(),
        ]
    )

    return smodel


def make_merged_LSTM_singular_model(
    model, hoyer=None, orthogonal=None, return_sequences=False
):
    smodel = keras.models.Sequential()

    smodel.add(
        keras.layers.InputLayer(input_shape=[None, model.input_shape[-1]])
    )

    for i, layer in enumerate(model.layers[:-1]):
        w, u, b = layer.get_weights()
        units = layer.units

        wu = np.concatenate((w, u), axis=0)

        left, sigma, right = np.linalg.svd(
            wu, full_matrices=False, compute_uv=True
        )

        sigma = np.expand_dims(sigma, axis=0)

        weights = (sigma, left, right, b)

        if hoyer is not None and hoyer != 0:
            kernel_regularizer = HoyerRegularizer(hoyer)
            recurrent_regularizer = HoyerRegularizer(hoyer)
        else:
            kernel_regularizer = None
            recurrent_regularizer = None
        if orthogonal is not None and orthogonal != 0:
            uv_regularizer = keras.regularizers.OrthogonalRegularizer(
                factor=orthogonal, mode="rows"
            )
            train_uv = True
        else:
            uv_regularizer = None
            train_uv = False

        cell = SingularLSTMCell(
            units,
            weights,
            kernel_type=2,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            train_uv=train_uv,
            uv_regularizer=uv_regularizer,
        )
        rs = True
        if i == len(model.layers[:-1]) - 1 and not return_sequences:
            rs = False
        lstm = SingularLSTM(units, cell=cell, return_sequences=rs)
        smodel.add(lstm)

    dense_top = keras.layers.Dense(1)

    if return_sequences:
        dense_top = keras.layers.TimeDistributed(dense_top)

    smodel.add(dense_top)
    dense_top.set_weights(
        [
            model.layers[-1].weights[0].numpy(),
            model.layers[-1].weights[1].numpy(),
        ]
    )

    return smodel


"""
Returns a singular model from the pretrained input model
hoyer: the coefficient of the Hoyer regularizer. if None or 0 no regularizer
is applied
orthogonal: the coefficient of the orthogonal regularizer. if None or 0
no regularizer is applied and U, V matrices are not trainable
"""


def make_LSTM_singular_model(
    model,
    hoyer=None,
    orthogonal=None,
    kernel_type=2,
    return_sequences=False,
):
    if kernel_type == 0:  # split kernel
        return make_split_LSTM_singular_model(
            model, hoyer=hoyer, return_sequences=return_sequences
        )
    elif kernel_type == 1:  # half merged kernel
        return make_half_merged_LSTM_singular_model(
            model,
            hoyer=hoyer,
            orthogonal=orthogonal,
            return_sequences=return_sequences,
        )
    else:  # merged kernel
        return make_merged_LSTM_singular_model(
            model,
            hoyer=hoyer,
            orthogonal=orthogonal,
            return_sequences=return_sequences,
        )


"""
From a model of SingularLSTMCells we create the reduced model. Cutoff
represents the maximum magnitude of singular values to be pruned. If
keep_sigmas is provided, then it is a list of nparray boolean masks which show
which sigmas to keep.
"""


def make_LSTM_reduced_model(
    model, keep_sigmas=None, cutoff=0.05, kernel_type=2
):
    rmodel = keras.models.Sequential()
    rmodel.add(
        keras.layers.InputLayer(input_shape=[None, model.input_shape[-1]])
    )
    if kernel_type == 1:
        for i, layer in enumerate(model.layers[:-1]):
            units = layer.units
            weights = layer.get_weights()

            w_l = weights[2]
            w_s = weights[0]
            w_r = weights[3]
            u_l = weights[4]
            u_s = weights[1]
            u_r = weights[5]
            b = weights[6]

            wu = []
            for j, mat in enumerate([[w_l, w_s, w_r], [u_l, u_s, u_r]]):
                U = mat[0]
                S = mat[1]
                V = mat[2]
                if keep_sigmas is not None:
                    indices = keep_sigmas[i][j]
                else:
                    indices = (S > cutoff)[0]
                U = U.T[indices[0]].T
                V = V[indices[0]]
                S = S[indices]
                r = V.shape[0]
                V1 = V[:, :r]
                V2 = V[:, r:]
                B = (U * S) @ V1
                C = np.linalg.inv(V1) @ V2
                wu.append([B, C])

            cell = ReducedLSTMCell(units, w=wu[0], u=wu[1], b=b, kernel_type=1)
            lstm = SingularLSTM(units, cell=cell, return_sequences=True)
            rmodel.add(lstm)
    elif kernel_type == 0:
        for i, layer in enumerate(model.layers[:-1]):
            w = []
            u = []
            units = layer.units
            weights = layer.get_weights()

            w_l = weights[2]
            w_s = weights[0]
            w_r = weights[3]
            u_l = weights[4]
            u_s = weights[1]
            u_r = weights[5]
            b = weights[6]

            w_l = np.split(w_l, 4, axis=1)
            w_s = np.split(w_s, 4, axis=1)
            w_r = np.split(w_r, 4, axis=1)
            u_l = np.split(u_l, 4, axis=1)
            u_s = np.split(u_s, 4, axis=1)
            u_r = np.split(u_r, 4, axis=1)

            for j in range(4):
                wu = []
                for k, mat in enumerate(
                    [
                        [w_l[j], w_s[j], w_r[j]],
                        [u_l[j], u_s[j], u_r[j]],
                    ]
                ):
                    U = mat[0]
                    S = mat[1]
                    V = mat[2]
                    if keep_sigmas is not None:
                        indices = keep_sigmas[i][k][j].reshape(1, -1)
                    else:
                        indices = (S > cutoff)[0]
                    U = U.T[indices[0]].T
                    V = V[indices[0]]
                    S = S[indices]
                    r = V.shape[0]
                    V1 = V[:, :r]
                    V2 = V[:, r:]
                    B = (U * S) @ V1
                    C = np.linalg.inv(V1) @ V2
                    wu.append([B, C])
                w.append(wu[0])
                u.append(wu[1])

            cell = ReducedLSTMCell(units, w=w, u=u, b=b, kernel_type=0)
            lstm = SingularLSTM(units, cell=cell, return_sequences=True)
            rmodel.add(lstm)
    else:  # merged kernel/kernel_type=2
        for i, layer in enumerate(model.layers[:-1]):
            units = layer.units
            weights = layer.get_weights()
            kernel = weights[0]
            w_left = weights[1]
            w_right = weights[2]
            b = weights[3]
            if keep_sigmas is not None:
                indices = keep_sigmas[i].reshape(1, -1)
            else:
                indices = (kernel > cutoff)[0]
            w_left = w_left.T[indices[0]].T
            w_right = w_right[indices[0]]
            kernel = kernel[indices]
            r = w_right.shape[0]
            V1 = w_right[:, :r]
            V2 = w_right[:, r:]
            B = (w_left * kernel) @ V1
            C = np.linalg.inv(V1) @ V2

            cell = ReducedLSTMCell(units, w=[B, C], b=b, kernel_type=2)
            lstm = SingularLSTM(units, cell=cell, return_sequences=True)
            rmodel.add(lstm)

    dense_top = keras.layers.TimeDistributed(keras.layers.Dense(1))
    rmodel.add(dense_top)
    dense_top.set_weights(
        [
            model.layers[-1].weights[0].numpy(),
            model.layers[-1].weights[1].numpy(),
        ]
    )
    return rmodel
