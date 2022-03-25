from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

import tensorflow.keras.layers as tkl
import tensorflow as tf

class MaskedDense(tkl.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_kernel=True,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):


        super(MaskedDense, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_kernel = use_kernel
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        if self.use_kernel:
            self.kernel = self.add_weight(name='kernel',
                                          shape=(self.units, last_dim),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        else:
            self.kernel = None

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)

        else:
            self.bias = None

        self.built = True

    def call(self, x, training=None, **kwargs):
        """ 
        x: (B, D)
        """

        batchsize = tf.shape(x)[0]
        output = tf.zeros((batchsize, self.units))
        if self.use_kernel:
            output = output + tf.squeeze(tf.matmul(self.kernel, tf.expand_dims(x, axis=-1)),
                                         axis=-1)
        if self.use_bias:
            output = self.activation(output + self.bias)
        return output


class ParallelDense(tkl.Layer):
    def __init__(self,
                 units,
                 T,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):


        super(ParallelDense, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.T = T
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self.kernels = []
        last_dim = input_shape[-1]
        for i in range(self.T):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(1, self.units, last_dim),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
        # (T, U, I)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.T, self.units),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)

        else:
            self.bias = None

        self.built = True




    def call(self, inputs, training=None, **kwargs):
        """ Inputs of size (B, T, I)"""
        kernel = tf.concat(self.kernels, axis=0)
        # (B, T, U)
        result = tf.squeeze(tf.matmul(kernel, tf.expand_dims(inputs, axis=-1)), axis=-1)
        if self.use_bias:
            # (B, T, U)
            result = result + self.bias

        return self.activation(result)

class ImbalancedParallelDense(tkl.Layer):
    def __init__(self,
                 T,
                 frequencies,
                 activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):


        super(ImbalancedParallelDense, self).__init__(**kwargs)
        self.T = T
        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get('glorot_uniform')
        if frequencies is not None:
            self.bias = tf.Variable(initial_value=-tf.math.log((1 - frequencies) / frequencies),
                                    trainable=True)
        else:
            self.bias = tf.Variable(initial_value=tf.zeros((self.T, )),
                                    trainable=True)

    def build(self, input_shape):
        self.kernels = []
        last_dim = input_shape[-1]
        for i in range(self.T):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(1, 1, last_dim),
                                                trainable=True))

        self.built = True


    def call(self, inputs, training=None, **kwargs):
        """ Inputs of size (B, T, I)"""
        # (T, 1, I)
        W = tf.concat(self.kernels, axis=0)
        # (B, T, 1)
        WX = tf.squeeze(tf.matmul(W, tf.expand_dims(inputs, axis=-1)), axis=-1)
        # (1, T, 1)
        b = self.bias[tf.newaxis, :, tf.newaxis]

        return self.activation(WX + b)


if __name__ == '__main__':
    parallel_dense1 = ParallelDense(units=256,
                                    T=6,
                                    activation='linear')

    parallel_dense2 = ParallelDense(units=128,
                                    T=6,
                                    activation='linear')
    a = tf.ones((12, 6, 128))
    tf.print(parallel_dense1(parallel_dense2(a)).shape)


