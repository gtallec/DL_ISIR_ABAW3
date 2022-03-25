import tensorflow.keras.layers as tkl
import tensorflow as tf

from tensorflow.python.util import nest
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from custom_ops import batch_matmul

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin

class GruCellForTracking(tkl.GRUCell):   
  def call(self, inputs, states, training=None):
    rec_output_dict = dict()
    h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(self.bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = K.dot(inputs_z, self.kernel[:, :self.units])
      x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          self.recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            self.recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs, self.kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)

      x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

      recurrent_z, recurrent_r, recurrent_h = array_ops.split(
          matrix_inner, [self.units, self.units, -1], axis=-1)

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1,
                            self.recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    rec_output_dict['z'] = z
    rec_output_dict['r'] = r
    new_state = [h] if nest.is_sequence(states) else h
    return h, new_state, rec_output_dict

class PermutationGRUCell(DropoutRNNCellMixin, tkl.Layer):
    def __init__(self,
                 units,
                 n_task,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 recurrent_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=0,
                 reset_after=False,
                 **kwargs):

        super(PermutationGRUCell, self).__init__(**kwargs)
        self.n_task = n_task
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            self.implementation = 1
        else:
            self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernels = []
        self.recurrent_kernels = []
        for i in range(self.n_task):
            self.kernels.append(self.add_weight(shape=(1, self.units * 3, input_dim),
                                                name="kernel_{}".format(i),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
            self.recurrent_kernels.append(self.add_weight(shape=(1, self.units * 3, self.units),
                                                          name="recurrent_kernel_{}".format(i),
                                                          initializer=self.kernel_initializer,
                                                          regularizer=self.kernel_regularizer,
                                                          constraint=self.kernel_constraint,
                                                          trainable=True))


        if self.use_bias:
            if not self.reset_after:
                bias_shape = (self.n_task, 3 * self.units)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, self.n_task, 3 * self.units)

            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, permutation, training=None, **kwargs):
        """
        permutation : (P,)
        inputs : (batch_size, P, I)
        states : (batch_size, P, H)
        """

        z_units = tf.range(0, self.units)
        r_units = tf.range(self.units, 2 * self.units)
        h_units = tf.range(2 * self.units, 3 * self.units)
        
        # (P, I, 3 * H)
        kernel = tf.concat(self.kernels, axis=0)
        recurrent_kernel = tf.concat(self.recurrent_kernels, axis=0)

        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        batch_size = tf.shape(inputs)[:-2]
        P = tf.shape(permutation)[0]

        # (P, I, 3 * H)
        permutated_recurrent_kernel = tf.gather(recurrent_kernel, permutation, axis=0)

        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)
                recurrent_bias = tf.gather(recurrent_bias, permutation, axis=0)
                recurrent_bias = tf.reshape(recurrent_bias, (*tf.ones_like(batch_size), P, 3 * self.units)) 
                recurrent_bias = tf.tile(recurrent_bias, multiples=[*batch_size, 1, 1])

            input_bias = tf.gather(input_bias, permutation, axis=0)

            input_bias = tf.reshape(input_bias, tf.concat([tf.ones_like(batch_size), [P, 3 * self.units]], axis=0))
            input_bias = tf.tile(input_bias, multiples=tf.concat([batch_size, [1, 1]], axis=0))

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            # (batch_size, P, U)
            x_z = batch_matmul(tf.gather(permutated_kernel, z_units, axis=-2),
                               inputs_z)
            # (P, batch_size, U) 
            x_r = batch_matmul(tf.gather(permutated_kernel, r_units, axis=-2),
                               inputs_r)
            # (P, batch_size, U)
            x_h = batch_matmul(tf.gather(permutated_kernel, h_units, axis=-2),
                               inputs_h)

            if self.use_bias:
                # (batch_size, P, U)
                x_z = x_z + tf.gather(input_bias, z_units, axis=-1)
                x_r = x_r + tf.gather(input_bias, r_units, axis=-1)
                x_h = x_h + tf.gather(input_bias, h_units, axis=-1)

            if 0. < self.recurrent_dropout < 1.:
                # (P, batch_size, H)
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                # (P, batch_size, H)
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            # (P, batch_size, H)
            recurrent_z = batch_matmul(tf.gather(permutated_recurrent_kernel, z_units, axis=-2),
                                       h_tm1_z)

            recurrent_r = batch_matmul(tf.gather(permutated_recurrent_kernel, r_units, axis=-2),
                                       h_tm1_r)

            if self.reset_after and self.use_bias:
                recurrent_z = recurrent_z + tf.gather(recurrent_bias, z_units, axis=-1)
                recurrent_r = recurrent_r + tf.gather(recurrent_bias, r_units, axis=-1)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                           h_tm1_h)
                if self.use_bias:
                    recurrent_h = recurrent_h + recurrent_bias[self.units * 2:]
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                           r * h_tm1_h)
            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                # (batch_size, P, 3 * H)
                inputs = inputs * dp_mask[0]
            # inputs projected by all gate matrices at once
            # (batchsize, P,  3 * H)
            inputs = batch_matmul(permutated_kernel, inputs)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                # (P, batch_size, 3 * H)
                inputs = inputs + input_bias

            tile_shape = tf.concat([tf.ones_like(batch_size), [1, 3]], axis=0)
            mask_shape = tf.concat([batch_size, [P, self.units]], axis=0)

            # [rec_z, rec_r, rec_h] : (B, P, 3 * H)
            input_h = batch_matmul(permutated_recurrent_kernel, h_tm1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                if self.use_bias:
                    input_h = input_h + recurrent_bias
 
            else:
                # hidden state projected separately for update/reset and new
                # [rec_z, rec_r, h_tm1] : (B, P, 3 * H)
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                           +
                           tf.tile(h_tm1, tile_shape) * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            #(B, P, U)

            # [z, r, rec_h or h_tm1] : (B, P, 3 * H)
            input_h = (self.recurrent_activation(((inputs + input_h) * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1))
                       +
                       input_h * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1)))

            if self.reset_after:
                # [z, r, r * rec_h]
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1) +
                           tf.tile(tf.gather(input_h, r_units, axis=-1), multiples=tile_shape) * input_h * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            else:

                # [z, r, r * (W_h x h_tm1)]
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                           +
                           tf.tile(batch_matmul(permutated_recurrent_kernel[:, 2 * self.units:, :],
                                                tf.gather(input_h, h_units, axis=-1)
                                                *
                                                tf.gather(input_h, r_units, axis=-1)), multiples=tile_shape)
                           *
                           tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                       +
                       self.recurrent_activation(inputs + input_h) * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))
        # previous and candidate state mixed by update gate
        h_tm1 = tf.gather(input_h, z_units, axis=-1) * h_tm1 + (1 - tf.gather(input_h, z_units, axis=-1)) * tf.gather(input_h, h_units, axis=-1)

        return h_tm1, [h_tm1] if nest.is_sequence(states) else h_tm1

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'recurrent_activation':
            activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'implementation': self.implementation,
            'reset_after': self.reset_after
        }
        base_config = super(PermutationGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class PermutationGRUCellv2(DropoutRNNCellMixin, tkl.Layer):
    def __init__(self,
                 units,
                 n_task,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 recurrent_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=0,
                 reset_after=False,
                 **kwargs):

        super(PermutationGRUCellv2, self).__init__(**kwargs)
        self.n_task = n_task
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            self.implementation = 1
        else:
            self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernels = []
        self.recurrent_kernels = []
        for i in range(self.n_task):
            self.kernels.append(self.add_weight(shape=(1, self.units * 3, input_dim),
                                                name="kernel_{}".format(i),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
            self.recurrent_kernels.append(self.add_weight(shape=(1, self.units * 3, self.units),
                                                          name="recurrent_kernel_{}".format(i),
                                                          initializer=self.kernel_initializer,
                                                          regularizer=self.kernel_regularizer,
                                                          constraint=self.kernel_constraint,
                                                          trainable=True))


        if self.use_bias:
            if not self.reset_after:
                bias_shape = (self.n_task, 3 * self.units)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, self.n_task, 3 * self.units)

            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True

    def transfer_kernels(self, i, j):
        kernel = self.kernels[i]
        recurrent_kernel = self.recurrent_kernels[i]
        self.kernels[j].assign(kernel)
        self.recurrent_kernels[j].assign(recurrent_kernel)

    def transfer_bias(self, i, j):
        # (T, )
        onehot = tf.eye(self.n_task)[j, :]
        # (3U)
        bias_i = self.bias[i, :]
        # (T, 3U)
        bias_i_tiled = tf.tile(tf.expand_dims(bias_i, axis=0), (self.n_task, 1))
        # (T, 3U)
        onehot_tiled = tf.tile(tf.expand_dims(onehot, axis=-1), (1, 3 * self.units))

        self.bias.assign(self.bias * (1 - onehot_tiled) + bias_i_tiled * onehot_tiled)



    def call(self, inputs, states, permutation, training=None, **kwargs):
        """
        permutation : (B, P)
        inputs : (B, P, I)
        states : (B, P, H)
        """

        z_units = tf.range(0, self.units)
        r_units = tf.range(self.units, 2 * self.units)
        h_units = tf.range(2 * self.units, 3 * self.units)
        
        # (T, 3H, I)
        kernel = tf.concat(self.kernels, axis=0)

        # (T, 3H, H)
        recurrent_kernel = tf.concat(self.recurrent_kernels, axis=0)

        # (B, P, 3H, H)
        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        batch_size = tf.shape(inputs)[:-2]
        P = tf.shape(permutation)[1]

        # (B, P, 3H, I)
        permutated_recurrent_kernel = tf.gather(recurrent_kernel, permutation, axis=0)

        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)
                # (B, P, 3H)
                recurrent_bias = tf.gather(recurrent_bias, permutation, axis=0)

                # recurrent_bias = tf.reshape(recurrent_bias, tf.concat([batch_size, [P, 3 * self.units]], axis=0)) 
                # recurrent_bias = tf.tile(recurrent_bias, multiples=[*batch_size, 1, 1])

            # (B, P, 3H)
            input_bias = tf.gather(input_bias, permutation, axis=0)

            # input_bias = tf.reshape(input_bias, tf.concat([tf.ones_like(batch_size), [P, 3 * self.units]], axis=0))
            # input_bias = tf.tile(input_bias, multiples=tf.concat([batch_size, [1, 1]], axis=0))

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            # (batch_size, P, U)
            x_z = batch_matmul(tf.gather(permutated_kernel, z_units, axis=-2),
                               inputs_z)
            # (P, batch_size, U) 
            x_r = batch_matmul(tf.gather(permutated_kernel, r_units, axis=-2),
                               inputs_r)
            # (P, batch_size, U)
            x_h = batch_matmul(tf.gather(permutated_kernel, h_units, axis=-2),
                               inputs_h)

            if self.use_bias:
                # (batch_size, P, U)
                x_z = x_z + tf.gather(input_bias, z_units, axis=-1)
                x_r = x_r + tf.gather(input_bias, r_units, axis=-1)
                x_h = x_h + tf.gather(input_bias, h_units, axis=-1)

            if 0. < self.recurrent_dropout < 1.:
                # (P, batch_size, H)
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                # (P, batch_size, H)
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            # (P, batch_size, H)
            recurrent_z = batch_matmul(tf.gather(permutated_recurrent_kernel, z_units, axis=-2),
                                       h_tm1_z)

            recurrent_r = batch_matmul(tf.gather(permutated_recurrent_kernel, r_units, axis=-2),
                                       h_tm1_r)

            if self.reset_after and self.use_bias:
                recurrent_z = recurrent_z + tf.gather(recurrent_bias, z_units, axis=-1)
                recurrent_r = recurrent_r + tf.gather(recurrent_bias, r_units, axis=-1)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                           h_tm1_h)
                if self.use_bias:
                    recurrent_h = recurrent_h + recurrent_bias[self.units * 2:]
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                           r * h_tm1_h)
            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                # (batch_size, P, I)
                inputs = inputs * dp_mask[0]
            # inputs projected by all gate matrices at once
            # (batchsize, P,  3 * H, 1)
            inputs = batch_matmul(permutated_kernel, inputs)

            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                # (P, batch_size, 3 * H)
                inputs = inputs + input_bias

            tile_shape = tf.concat([tf.ones_like(batch_size), [1, 3]], axis=0)
            mask_shape = tf.concat([batch_size, [P, self.units]], axis=0)

            # [rec_z, rec_r, rec_h] : (B, P, 3 * H)           
            input_h = batch_matmul(permutated_recurrent_kernel, h_tm1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                if self.use_bias:
                    input_h = input_h + recurrent_bias
 
            else:
                # hidden state projected separately for update/reset and new
                # [rec_z, rec_r, h_tm1] : (B, P, 3 * H)
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                           +
                           tf.tile(h_tm1, tile_shape) * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            # [z, r, rec_h or h_tm1] : (B, P, 3 * H)
            input_h = (self.recurrent_activation(((inputs + input_h) * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1))
                       +
                       input_h * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1)))

            if self.reset_after:
                # [z, r, r * rec_h]
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1) +
                           tf.tile(tf.gather(input_h, r_units, axis=-1), multiples=tile_shape) * input_h * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            else:

                # [z, r, r * (W_h x h_tm1)]
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                           +
                           tf.tile(batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                                tf.gather(input_h, h_units, axis=-1)
                                                *
                                                tf.gather(input_h, r_units, axis=-1)), multiples=tile_shape)
                           *
                           tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                       +
                       self.recurrent_activation(inputs + input_h) * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))
        # previous and candidate state mixed by update gate
        h_tm1 = tf.gather(input_h, z_units, axis=-1) * h_tm1 + (1 - tf.gather(input_h, z_units, axis=-1)) * tf.gather(input_h, h_units, axis=-1)

        return h_tm1, [h_tm1] if nest.is_sequence(states) else h_tm1

class PermutationGRUCellv3(DropoutRNNCellMixin, tkl.Layer):
    def __init__(self,
                 units,
                 n_task,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 recurrent_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=0,
                 reset_after=False,
                 **kwargs):

        super(PermutationGRUCellv3, self).__init__(**kwargs)
        self.n_task = n_task
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            self.implementation = 1
        else:
            self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernels = []
        self.recurrent_kernels = []
        for i in range(self.n_task):
            self.kernels.append(self.add_weight(shape=(1, self.units * 3, input_dim),
                                                name="kernel_{}".format(i),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
            self.recurrent_kernels.append(self.add_weight(shape=(1, self.units * 3, self.units),
                                                          name="recurrent_kernel_{}".format(i),
                                                          initializer=self.kernel_initializer,
                                                          regularizer=self.kernel_regularizer,
                                                          constraint=self.kernel_constraint,
                                                          trainable=True))


        if self.use_bias:
            if not self.reset_after:
                bias_shape = (self.n_task, 3 * self.units)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, self.n_task, 3 * self.units)

            self.bias = self.add_weight(shape=bias_shape,
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, permutation, mask, training=None, **kwargs):
        """
        permutation (B, P): Tasks to process in parallel for each element. 
        mask (B, P): 1 if the task should be skipped in train, 0 else. 
        inputs : (B, P, I)
        states : (B, P, H)
        """

        z_units = tf.range(0, self.units)
        r_units = tf.range(self.units, 2 * self.units)
        h_units = tf.range(2 * self.units, 3 * self.units)
        
        # (T, 3H, I)
        kernel = tf.concat(self.kernels, axis=0)

        # (T, 3H, H)
        recurrent_kernel = tf.concat(self.recurrent_kernels, axis=0)

        # (B, P, 3H, H)
        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        batch_size = tf.shape(inputs)[0]
        P = tf.shape(permutation)[1]

        # (B, P, 3H, I)
        permutated_recurrent_kernel = tf.gather(recurrent_kernel, permutation, axis=0)

        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=3)
        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)
                # (B, P, 3H)
                recurrent_bias = tf.gather(recurrent_bias, permutation, axis=0)

                # recurrent_bias = tf.reshape(recurrent_bias, tf.concat([batch_size, [P, 3 * self.units]], axis=0)) 
                # recurrent_bias = tf.tile(recurrent_bias, multiples=[*batch_size, 1, 1])

            # (B, P, 3H)
            input_bias = tf.gather(input_bias, permutation, axis=0)

            # input_bias = tf.reshape(input_bias, tf.concat([tf.ones_like(batch_size), [P, 3 * self.units]], axis=0))
            # input_bias = tf.tile(input_bias, multiples=tf.concat([batch_size, [1, 1]], axis=0))

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            # (batch_size, P, U)
            x_z = batch_matmul(tf.gather(permutated_kernel, z_units, axis=-2),
                               inputs_z)
            # (P, batch_size, U) 
            x_r = batch_matmul(tf.gather(permutated_kernel, r_units, axis=-2),
                               inputs_r)
            # (P, batch_size, U)
            x_h = batch_matmul(tf.gather(permutated_kernel, h_units, axis=-2),
                               inputs_h)

            if self.use_bias:
                # (batch_size, P, U)
                x_z = x_z + tf.gather(input_bias, z_units, axis=-1)
                x_r = x_r + tf.gather(input_bias, r_units, axis=-1)
                x_h = x_h + tf.gather(input_bias, h_units, axis=-1)

            if 0. < self.recurrent_dropout < 1.:
                # (P, batch_size, H)
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                # (P, batch_size, H)
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            # (P, batch_size, H)
            recurrent_z = batch_matmul(tf.gather(permutated_recurrent_kernel, z_units, axis=-2),
                                       h_tm1_z)

            recurrent_r = batch_matmul(tf.gather(permutated_recurrent_kernel, r_units, axis=-2),
                                       h_tm1_r)

            if self.reset_after and self.use_bias:
                recurrent_z = recurrent_z + tf.gather(recurrent_bias, z_units, axis=-1)
                recurrent_r = recurrent_r + tf.gather(recurrent_bias, r_units, axis=-1)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                           h_tm1_h)
                if self.use_bias:
                    recurrent_h = recurrent_h + recurrent_bias[self.units * 2:]
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                           r * h_tm1_h)
            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                # (batch_size, P, I)
                inputs = inputs * dp_mask[0]
            # inputs projected by all gate matrices at once
            # (batchsize, P,  3 * H, 1)
            inputs = batch_matmul(permutated_kernel, inputs)

            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                # (P, batch_size, 3 * H)
                inputs = inputs + input_bias

            tile_shape = [1, 1, 3]
            mask_shape = [batch_size, P, self.units]

            # [rec_z, rec_r, rec_h] : (B, P, 3 * H)           
            input_h = batch_matmul(permutated_recurrent_kernel, h_tm1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                if self.use_bias:
                    input_h = input_h + recurrent_bias
 
            else:
                # hidden state projected separately for update/reset and new
                # [rec_z, rec_r, h_tm1] : (B, P, 3 * H)
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                           +
                           tf.tile(h_tm1, tile_shape) * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            # [z, r, rec_h or h_tm1] : (B, P, 3 * H)
            input_h = (self.recurrent_activation(((inputs + input_h) * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1))
                       +
                       input_h * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1)))

            if self.reset_after:
                # [z, r, r * rec_h]
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1) +
                           tf.tile(tf.gather(input_h, r_units, axis=-1), multiples=tile_shape) * input_h * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            else:

                # [z, r, r * (W_h x h_tm1)]
                input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                           +
                           tf.tile(batch_matmul(tf.gather(permutated_recurrent_kernel, h_units, axis=-2),
                                                tf.gather(input_h, h_units, axis=-1)
                                                *
                                                tf.gather(input_h, r_units, axis=-1)), multiples=tile_shape)
                           *
                           tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))

            input_h = (input_h * tf.concat([tf.ones(mask_shape), tf.ones(mask_shape), tf.zeros(mask_shape)], axis=-1)
                       +
                       self.recurrent_activation(inputs + input_h) * tf.concat([tf.zeros(mask_shape), tf.zeros(mask_shape), tf.ones(mask_shape)], axis=-1))
        # previous and candidate state mixed by update gate
        h_t = tf.gather(input_h, z_units, axis=-1) * h_tm1 + (1 - tf.gather(input_h, z_units, axis=-1)) * tf.gather(input_h, h_units, axis=-1)
        # mask tasks that are partially labeled
        tiled_label_mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, self.units])
        h_t = h_t * tiled_label_mask + h_tm1 * (1 - tiled_label_mask) 

        return h_t, [h_t] if nest.is_sequence(states) else h_t

class PermutationRNNCell(DropoutRNNCellMixin, tkl.Layer):
    def __init__(self,
                 n_task,
                 units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):

        super(PermutationRNNCell, self).__init__(**kwargs)
        self.n_task = n_task
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        self.kernels = []
        self.recurrent_kernels = []
        for i in range(self.n_task):
            self.kernels.append(self.add_weight(shape=(1, self.units, input_shape[-1]),
                                                name='kernel_{}'.format(i),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint))
            self.recurrent_kernels.append(self.add_weight(shape=(1, self.units, self.units),
                                                          name='recurrent_kernel_{}'.format(i),
                                                          initializer=self.recurrent_initializer,
                                                          regularizer=self.recurrent_regularizer,
                                                          constraint=self.recurrent_constraint))
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.n_task, self.units),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True


    def call(self, inputs, states, permutation, training=None, **kwargs):
        """ 
        permutation : (B, P)
        inputs : (batch_size, P, I)
        states : (batch_size, P, H)
        """

        # (T, H, I) 
        kernel = tf.concat(self.kernels, axis=0)

        # (T, H, H)
        recurrent_kernel = tf.concat(self.recurrent_kernels, axis=0)
        batch_size = tf.shape(inputs)[0]

        # (B, P, H, I)
        kernel = tf.gather(kernel, permutation, axis=0)
        # (B, P, H, H)
        recurrent_kernel = tf.gather(recurrent_kernel, permutation, axis=0)

        h_tm1 = states[0] if nest.is_sequence(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training)

        if dp_mask is not None:
            inputs = inputs * dp_mask

        # (B, P, H)
        inputs = batch_matmul(kernel, inputs)

        if self.bias is not None:
            # (B, P, H)
            bias = tf.gather(self.bias, permutation, axis=0)
            # (B, P, H)
            inputs = inputs + bias

        if rec_dp_mask is not None:
            h_tm1 = h_tm1 * rec_dp_mask

        h_tm1 = inputs + batch_matmul(recurrent_kernel, h_tm1)

        if self.activation is not None:
            h_tm1 = self.activation(h_tm1)

        return h_tm1, [h_tm1] if nest.is_sequence(states) else h_tm1


SUPPORTED_CELLS = {'rnn': tkl.SimpleRNNCell,
                   'gru': tkl.GRUCell,
                   'permutation_gru': PermutationGRUCell,
                   'permutation_rnn': PermutationRNNCell,
                   'permutation_gruv2': PermutationGRUCellv2,
                   'permutation_gruv3': PermutationGRUCellv3,
                   'track_gru': GruCellForTracking}

SUPPORTED_TRACKED_CELLS = {'gru': GruCellForTracking}

def tracked_recurrent_cell(recurrent_cell_args):
    recurrent_cell_type = recurrent_cell_args.pop('type')
    return SUPPORTED_TRACKED_CELLS[recurrent_cell_type](**recurrent_cell_args)

def recurrent_cell(recurrent_cell_args):
    recurrent_cell_type = recurrent_cell_args.pop('type')
    return SUPPORTED_CELLS[recurrent_cell_type](**recurrent_cell_args)

if __name__ == '__main__':
    pass



    




