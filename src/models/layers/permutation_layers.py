from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

import tensorflow.keras.layers as tkl
import tensorflow as tf

from custom_ops import batch_matmul


class BlockPermutationDense(tkl.Layer):
    def __init__(self,
                 blocks,
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
        super(BlockPermutationDense, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.blocks = blocks 

        self.n_blocks = len(blocks)
        self.len_blocks = [len(block) for block in blocks]
        self.n_task = sum(self.len_blocks)

        self.activation = activations.get(activation)
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
        for i in range(self.n_blocks):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(self.len_blocks[i], last_dim),
                                                initializer=self.kernel_initializer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.n_task,),
                                        initializer=self.bias_initializer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self,
             inputs,
             permutation,
             training=None,
             **kwargs):

        """
        inputs : (B, N, I),
        permutation : (N,)
        """

        N = tf.shape(permutation)[0]
        batch_size = tf.shape(inputs)[0]
        last_dim = tf.shape(inputs)[-1]

        coord_block = []
        permutated_kernel = []
    
        print(permutation[0])
        for i in range(N):
            coord_block.append(self.blocks[permutation[i]])
            permutated_kernel.append(self.kernels[permutation[i]])

        coord_block = sum(coord_block, [])
        n_coord = tf.shape(coord_block)[0]
        permutated_length = tf.gather(self.len_blocks, permutation, axis=0)
        permutated_bias = tf.gather(self.bias, coord_block, axis=0)

        kernel_matrix = []
        cursor = 0
        for i in range(N):
            block_length = permutated_length[i]
            kernel_matrix.append(tf.concat([tf.zeros((cursor, last_dim)),
                                            permutated_kernel[i],
                                            tf.zeros((n_coord - (cursor + block_length), last_dim))],
                                           axis=0))
            cursor = cursor + block_length


        # (N_C, N x I)
        kernel_matrix = tf.concat(kernel_matrix, axis=1)
        tf.print(kernel_matrix)
        # (B, N x I)
        inputs = tf.reshape(inputs, (batch_size, N * last_dim, 1))
        inputs = tf.squeeze(tf.matmul(kernel_matrix, inputs), axis=-1)

        if self.use_bias:
            inputs = inputs + permutated_bias

        return inputs

class PermutationDense(tkl.Layer):
    def __init__(self,
                 n_task,
                 units,
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
        super(PermutationDense, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.n_task = n_task
        self.activation = activations.get(activation)
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
        for i in range(self.n_task):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(1, self.units, last_dim),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.n_task, self.units),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, permutation, training=None, **kwargs):
        """ inputs of size (batch_size, P, I) 
            permutation : (batch_size, P) """
        batchsize = tf.shape(inputs)[:-2]
        kernel = tf.concat(self.kernels, axis=0)
        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        result = batch_matmul(permutated_kernel, inputs)

        if self.use_bias:
            bias = tf.gather(self.bias, permutation, axis=0)
            bias = tf.reshape(bias, tf.concat([tf.ones_like(batchsize), [tf.shape(permutation)[0], self.units]], axis=0))
            bias = tf.tile(bias, multiples=tf.concat([batchsize, [1, 1]], axis=0))
            result = result + bias
        return self.activation(result)


class PermutationDensev2(tkl.Layer):
    def __init__(self,
                 n_task,
                 units,
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
        super(PermutationDensev2, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.n_task = n_task
        self.activation = activations.get(activation)
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
        for i in range(self.n_task):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(1, self.units, last_dim),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.n_task, self.units),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, permutation, training=None, **kwargs):
        """ inputs of size (batch_size, P, I) 
            permutation : (batch_size, P) """
        kernel = tf.concat(self.kernels, axis=0)
        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        result = batch_matmul(permutated_kernel, inputs)

        if self.use_bias:
            bias = tf.gather(self.bias, permutation, axis=0)
            result = result + bias
        return self.activation(result)

class ImbalancedPermutationDense(tkl.Layer):
    def __init__(self,
                 T,
                 frequencies,
                 activation='sigmoid',
                 **kwargs):
        super(ImbalancedPermutationDense, self).__init__(**kwargs)
        self.T = T
        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get('glorot_uniform')
        self.bias = tf.Variable(initial_value=-tf.math.log((1 - frequencies) / frequencies),
                                trainable=True)

    def build(self, input_shape):
        self.kernels = []
        last_dim = input_shape[-1]
        for i in range(self.T):
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(1, 1, last_dim),
                                                initializer=self.kernel_initializer,
                                                trainable=True))
        self.built = True

    def transfer_kernels(self, i, j):
        kernel = self.kernels[i]
        self.kernels[j].assign(kernel)

    def transfer_bias(self, i, j):
        onehot = tf.eye(self.T)[j, :]
        bias_i = self.bias[i]
        self.bias.assign(self.bias * (1 - onehot) + bias_i * onehot)


    def call(self, inputs, permutation, training=None, **kwargs):
        """
        inputs of size (batch_size, P, I)
        permutation : (batch_size, P)
        """

        # (T, 1, I)
        kernel = tf.concat(self.kernels, axis=0)
        # (B, P, 1, I)
        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        # (B, P, 1)
        result = batch_matmul(permutated_kernel, inputs)
        # (B, P)
        bias = tf.expand_dims(tf.gather(self.bias, permutation, axis=0), axis=-1)
        result = result + bias
        return self.activation(result)

class PermutationDenseCategorical(tkl.Layer): 
    def __init__(self,
                 n_classes,
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
        super(PermutationDenseCategorical, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.n_classes = n_classes
        self.n_tasks = len(n_classes) 
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
 
    def build(self, input_shape):
        self.kernels = []
        self.kernel_masks = []
        last_dim = input_shape[-1]
        for i in range(len(self.n_classes)):
            # (SUM(Ti), last_dim) 
            kernel_mask_i = ([tf.zeros((self.n_classes[j], last_dim)) for j in range(i)] + 
                             [tf.ones((self.n_classes[i], last_dim))] + 
                             [tf.zeros((self.n_classes[j], last_dim)) for j in range(i + 1, len(self.n_classes))])
            self.kernel_masks.append(tf.expand_dims(tf.concat(kernel_mask_i, axis=0), axis=0))
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(self.n_classes[i], last_dim),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
        # (T, SUM(Ti), last_dim)
        self.kernel_masks = tf.concat(self.kernel_masks, axis=0)

        if self.use_bias:
            self.biases = []
            self.bias_masks = []
            for i in range(len(self.n_classes)):
                # (SUM(Ti), )
                bias_mask_i = ([tf.zeros((self.n_classes[j], )) for j in range(i)] + 
                               [tf.ones((self.n_classes[i], ))] + 
                               [tf.zeros((self.n_classes[j], )) for j in range(i + 1, len(self.n_classes))])
                self.bias_masks.append(tf.expand_dims(tf.concat(bias_mask_i, axis=0), axis=0))


                self.biases.append(self.add_weight(name='bias_{}'.format(i),
                                                   shape=(self.n_classes[i], ),
                                                   initializer=self.bias_initializer,
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint,
                                                   trainable=True))
            self.bias_masks = tf.concat(self.bias_masks, axis=0)

        self.built = True

    def call(self, inputs, permutation, training=None, **kwargs):
        """
        Ins: 
        inputs: (batch_size, P, I) 
        permutation: (batch_size, P)

        Outs:
        result: (batch_size, P, sum(Ti))
        task_mask: (batch_size, P, sum(Ti))
        """
        # (T, SUM(Ti), last_dim)
        kernel = tf.tile(tf.expand_dims(tf.concat(self.kernels, axis=0), axis=0), (self.n_tasks, 1, 1))
        # (T, SUM(Ti), last_dim)
        kernel = self.kernel_masks * kernel
        # (T, SUM(Ti))
        bias = tf.tile(tf.expand_dims(tf.concat(self.biases, axis=0), axis=0), (self.n_tasks, 1))
        # (T, SUM(Ti))
        bias = self.bias_masks * bias

        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        result = batch_matmul(permutated_kernel, inputs)

        if self.use_bias:
            bias = tf.gather(bias, permutation, axis=0)
            result = result + bias

        task_mask = tf.gather(self.bias_masks, permutation, axis=0)
        return self.activation(result), task_mask


class PermutationDenseCategoricalv2(tkl.Layer): 
    def __init__(self,
                 n_classes,
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
        super(PermutationDenseCategoricalv2, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.n_classes = n_classes
        self.n_tasks = len(n_classes) 
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
 
    def build(self, input_shape):
        self.kernels = []
        self.kernel_masks = []
        last_dim = input_shape[-1]
        for i in range(len(self.n_classes)):
            # (SUM(Ti), last_dim) 
            kernel_mask_i = ([tf.zeros((self.n_classes[j], last_dim)) for j in range(i)] + 
                             [tf.ones((self.n_classes[i], last_dim))] + 
                             [tf.zeros((self.n_classes[j], last_dim)) for j in range(i + 1, len(self.n_classes))])
            self.kernel_masks.append(tf.expand_dims(tf.concat(kernel_mask_i, axis=0), axis=0))
            self.kernels.append(self.add_weight(name='kernel_{}'.format(i),
                                                shape=(self.n_classes[i], last_dim),
                                                initializer=self.kernel_initializer,
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))
        # (T, SUM(Ti), last_dim)
        self.kernel_masks = tf.concat(self.kernel_masks, axis=0)

        if self.use_bias:
            self.biases = []
            self.bias_masks = []
            for i in range(len(self.n_classes)):
                # (SUM(Ti), )
                bias_mask_i = ([tf.zeros((self.n_classes[j], )) for j in range(i)] + 
                               [tf.ones((self.n_classes[i], ))] + 
                               [tf.zeros((self.n_classes[j], )) for j in range(i + 1, len(self.n_classes))])
                self.bias_masks.append(tf.expand_dims(tf.concat(bias_mask_i, axis=0), axis=0))


                self.biases.append(self.add_weight(name='bias_{}'.format(i),
                                                   shape=(self.n_classes[i], ),
                                                   initializer=self.bias_initializer,
                                                   regularizer=self.bias_regularizer,
                                                   constraint=self.bias_constraint,
                                                   trainable=True))
            self.bias_masks = tf.concat(self.bias_masks, axis=0)

        self.built = True

    def call(self, inputs, permutation, training=None, **kwargs):
        """
        Ins: 
        inputs: (batch_size, P, I) 
        permutation: (batch_size, P)

        Outs:
        result: (batch_size, P, sum(Ti))
        task_mask: (batch_size, P, sum(Ti))
        """
        # (T, SUM(Ti), last_dim)
        kernel = tf.tile(tf.expand_dims(tf.concat(self.kernels, axis=0), axis=0), (self.n_tasks, 1, 1))
        # (T, SUM(Ti), last_dim)
        kernel = self.kernel_masks * kernel

        # (T, SUM(Ti))
        bias = tf.tile(tf.expand_dims(tf.concat(self.biases, axis=0), axis=0), (self.n_tasks, 1))
        # (T, SUM(Ti))
        bias = self.bias_masks * bias

        permutated_kernel = tf.gather(kernel, permutation, axis=0)
        result = batch_matmul(permutated_kernel, inputs)

        if self.use_bias:
            bias = tf.gather(bias, permutation, axis=0)
            result = result + bias

        return self.activation(result)

if __name__ == '__main__':
    categorical_dense = PermutationDenseCategorical(n_classes=[4, 4])
    categorical_dense.build((None, 64))
