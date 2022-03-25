import models.regressors.permutation_regressors.permutation_heuristics as permutation_heuristics
import models.regressors.permutation_regressors.block_utils as block_utils
import models.regressors.permutation_regressors.utils as utils

from models.layers.recurrent_cells import recurrent_cell
from models.layers.permutation_layers import PermutationDense, PermutationDensev2, PermutationDenseCategorical, ImbalancedPermutationDense
from utils import sample_from_categorical, sample_without_replacement

import tensorflow.keras.layers as tkl
import tensorflow_probability as tfp
import tensorflow.keras.models as tkm
import tensorflow as tf
import numpy as np

import copy

class Monet(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 drop_out,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding='no',
                 permutation_units=64,
                 **kwargs):
        super(Monet, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        self.drop_out = drop_out

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        print(permutation_heuristic)
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        tf.print(permutations)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (n_permutations, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = n_task

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = PermutationDensev2(n_task=n_task,
                                        units=1,
                                        activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.previous_label_encoder = tkl.Dense(units=label_units,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)

        self.permutation_encoding = permutation_encoding
        if permutation_encoding != 'no': 
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)


        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(Monet, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        #########################
        # Mixture concerns

        abs_permutation_mixture_logits = self.vector(None)
        abs_permutation_mixture = tf.nn.softmax(abs_permutation_mixture_logits)

        n_permutations = self.n_permutations
        if training:
            # Sample permutations that are not dropped out for each example
            # (B, DO)
            permutation_sample_do = sample_without_replacement(tf.zeros((batch_size, self.n_permutations)),
                                                               self.drop_out)

            # (B, DO, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_sample_do, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_sample_do, axis=0)

            # (B, DO)
            permutation_mixture_logits = tf.gather(abs_permutation_mixture_logits, permutation_sample_do, axis=0)
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits, axis=1) 
            n_permutations = self.drop_out
        else:
            # (P, T, T)
            permutation_matrices = self.permutation_matrices
            permutation_mixture_logits = abs_permutation_mixture_logits
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)


        perm_expend = tf.reshape(abs_permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = abs_permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)
        
        outputs_dict['mixture_logits'] = tf.tile(tf.expand_dims(abs_permutation_mixture_logits,
                                                                axis=0),
                                                 (batch_size, 1))
       
        """
        for i in range(self.n_permutations):
            outputs_dict['mixture_logits_{}'.format(i)] = abs_permutation_mixture_logits[i]
        """

        #########################
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = n_permutations
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (B, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (B, P, T)
            y = tf.squeeze(tf.matmul(permutation_matrices, tf.expand_dims(y, axis=-1)), axis=-1)

            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, n_permutations, 1)), y], axis=-1)

        else:
            parallel = self.N_sample
            # (B, P)
            tiled_permutation_mixture_logits = tf.tile(tf.expand_dims(permutation_mixture_logits, axis=0),
                                                       (batch_size, 1))
            # (B, N)
            permutation_samples = tf.random.categorical(tiled_permutation_mixture_logits, self.N_sample)

            # (B, L, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_samples, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)


        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.permutation_encoding == 'perm_token':
            if training:
                kept_permutations = permutation_sample_do
            else:
                kept_permutations = permutation_samples

            # (P, P)
            permutation_onehots = tf.eye(self.n_permutations)

            # (B, k or L, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)



        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        # (B, ?, T + 1, T)
        padded_permutations = tf.concat([tf.zeros((batch_size, parallel, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=2)
        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, parallel))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)

            # (B, ?, 1)_
            y_k = tf.expand_dims(y_k, axis=-1)
            # (B, ?, T)
            y_k = tf.tile(y_k, multiples=[1, 1, self.n_task])

            # (B, ?, T)
            if k != 0:
                y_k = 2 * y_k - 1 

            # (1, ?, T)
            projection = padded_permutations[:, :, k, :]

            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, :, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, ?, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, ?)
                y_k = tf.squeeze(y_k, axis=-1)


        # (B, ?, T)
        logits = tf.concat(logits, axis=-1)
        # (B, ?, T, 1)
        logits = tf.expand_dims(logits, axis=-1)

        # (B, ?, T) 
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

        """
        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]
        """

        if training:

            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['prediction_logits'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (batch_size, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        pred = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        pred = tf.clip_by_value(pred, 0, 1)
        outputs_dict['global_pred'] = pred
        return outputs_dict


class HardMonet(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding='no',
                 permutation_units=64,
                 **kwargs):
        super(HardMonet, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (n_permutations, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = n_task

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = PermutationDensev2(n_task=n_task,
                                        units=1,
                                        activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.previous_label_encoder = tkl.Dense(units=label_units,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)

        self.permutation_encoding = permutation_encoding
        if permutation_encoding != 'no': 
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)


        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.vector = vector
        self.mixture_mask = tf.Variable(tf.ones((self.n_permutations, )), trainable=False) 

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(HardMonet, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        #########################
        # Mixture concerns

        permutation_mixture_logits = self.vector(None)
        permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        permutation_mixture = permutation_mixture * self.mixture_mask
        permutation_mixture = permutation_mixture / tf.math.reduce_sum(permutation_mixture)


        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)
        
        outputs_dict['mixture_logits'] = tf.tile(tf.expand_dims(permutation_mixture_logits,
                                                                axis=0),
                                                 (batch_size, 1))
       

        permutation_mixture = tf.tile(tf.reshape(permutation_mixture, (1, self.n_permutations)), (batch_size, 1))
        #########################
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = self.n_permutations
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (B, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (B, P, T)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, tf.expand_dims(y, axis=-1)), axis=-1)

            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)

            permutation_matrices = tf.tile(tf.expand_dims(self.permutation_matrices, axis=0), (batch_size, 1, 1, 1))
            permutation_mapping = tf.tile(tf.expand_dims(self.permutation_mapping, axis=0), (batch_size, 1, 1))

        else:
            parallel = self.N_sample
            # (B, P)
            tiled_permutation_mixture_logits = tf.tile(tf.expand_dims(permutation_mixture_logits, axis=0),
                                                       (batch_size, 1))
            # (B, N)
            permutation_samples = tf.random.categorical(tiled_permutation_mixture_logits, self.N_sample)

            # (B, L, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_samples, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)


        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.permutation_encoding == 'perm_token':
            if training:
                kept_permutations = tf.range(self.n_permutations)
            else:
                kept_permutations = permutation_samples

            # (P, P)
            permutation_onehots = tf.eye(self.n_permutations)

            # (B, k or L, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)



        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        # (B, ?, T + 1, T)
        padded_permutations = tf.concat([tf.zeros((batch_size, parallel, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=2)
        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, parallel))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)

            # (B, ?, 1)_
            y_k = tf.expand_dims(y_k, axis=-1)
            # (B, ?, T)
            y_k = tf.tile(y_k, multiples=[1, 1, self.n_task])

            # (B, ?, T)
            if k != 0:
                y_k = 2 * y_k - 1 

            # (1, ?, T)
            projection = padded_permutations[:, :, k, :]

            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, :, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, ?, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, ?)
                y_k = tf.squeeze(y_k, axis=-1)


        # (B, ?, T)
        logits = tf.concat(logits, axis=-1)
        # (B, ?, T, 1)
        logits = tf.expand_dims(logits, axis=-1)

        # (B, ?, T) 
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['prediction_logits'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (batch_size, self.n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        pred = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        pred = tf.clip_by_value(pred, 0, 1)
        outputs_dict['global_pred'] = pred
        return outputs_dict

    def change_mask(self, mask):
        self.mixture_mask.assign(tf.dtypes.cast(mask, tf.float32))

class SingleOrderMonet(tkm.Model):
    def __init__(self,
                 n_task,
                 order,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding='no',
                 permutation_units=64,
                 **kwargs):
        super(SingleOrderMonet, self).__init__()
        self.n_task = n_task
        self.n_permutations = 1
        self.N_sample = N_sample
        self.drop_out = 1

        permutations = tf.expand_dims(tf.constant(order), axis=0)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (n_permutations, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = n_task

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = PermutationDensev2(n_task=n_task,
                                        units=1,
                                        activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.previous_label_encoder = tkl.Dense(units=label_units,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)

        self.permutation_encoding = permutation_encoding
        if permutation_encoding != 'no': 
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)


        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(SingleOrderMonet, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        #########################
        # Mixture concerns

        abs_permutation_mixture_logits = self.vector(None)
        abs_permutation_mixture = tf.nn.softmax(abs_permutation_mixture_logits)

        n_permutations = self.n_permutations
        if training:
            # Sample permutations that are not dropped out for each example
            # (B, DO)
            permutation_sample_do = sample_without_replacement(tf.zeros((batch_size, self.n_permutations)),
                                                               self.drop_out)

            # (B, DO, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_sample_do, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_sample_do, axis=0)

            # (B, DO)
            permutation_mixture_logits = tf.gather(abs_permutation_mixture_logits, permutation_sample_do, axis=0)
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits, axis=1) 
            n_permutations = self.drop_out
        else:
            # (P, T, T)
            permutation_matrices = self.permutation_matrices
            permutation_mixture_logits = abs_permutation_mixture_logits
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)


        perm_expend = tf.reshape(abs_permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = abs_permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)
        
        outputs_dict['mixture_logits'] = tf.tile(tf.expand_dims(abs_permutation_mixture_logits,
                                                                axis=0),
                                                 (batch_size, 1))
       
        """
        for i in range(self.n_permutations):
            outputs_dict['mixture_logits_{}'.format(i)] = abs_permutation_mixture_logits[i]
        """

        #########################
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = n_permutations
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (B, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (B, P, T)
            y = tf.squeeze(tf.matmul(permutation_matrices, tf.expand_dims(y, axis=-1)), axis=-1)

            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, n_permutations, 1)), y], axis=-1)

        else:
            parallel = self.N_sample
            # (B, P)
            tiled_permutation_mixture_logits = tf.tile(tf.expand_dims(permutation_mixture_logits, axis=0),
                                                       (batch_size, 1))
            # (B, N)
            permutation_samples = tf.random.categorical(tiled_permutation_mixture_logits, self.N_sample)

            # (B, L, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_samples, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)


        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.permutation_encoding == 'perm_token':
            if training:
                kept_permutations = permutation_sample_do
            else:
                kept_permutations = permutation_samples

            # (P, P)
            permutation_onehots = tf.eye(self.n_permutations)

            # (B, k or L, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)



        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        # (B, ?, T + 1, T)
        padded_permutations = tf.concat([tf.zeros((batch_size, parallel, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=2)
        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, parallel))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)

            # (B, ?, 1)_
            y_k = tf.expand_dims(y_k, axis=-1)
            # (B, ?, T)
            y_k = tf.tile(y_k, multiples=[1, 1, self.n_task])

            # (B, ?, T)
            if k != 0:
                y_k = 2 * y_k - 1 

            # (1, ?, T)
            projection = padded_permutations[:, :, k, :]

            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, :, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, ?, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, ?)
                y_k = tf.squeeze(y_k, axis=-1)


        # (B, ?, T)
        logits = tf.concat(logits, axis=-1)
        # (B, ?, T, 1)
        logits = tf.expand_dims(logits, axis=-1)

        # (B, ?, T) 
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

        """
        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]
        """

        if training:

            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['prediction_logits'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (batch_size, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        pred = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        pred = tf.clip_by_value(pred, 0, 1)
        outputs_dict['global_pred'] = pred
        return outputs_dict


class ImbalancedMonet(tkm.Model):
    def __init__(self,
                 n_task,
                 frequencies,
                 n_permutations,
                 drop_out,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding='no',
                 permutation_units=64,
                 **kwargs):
        super(ImbalancedMonet, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        self.drop_out = drop_out

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (n_permutations, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = n_task

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = ImbalancedPermutationDense(T=n_task,
                                                frequencies=frequencies,
                                                activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.previous_label_encoder = tkl.Dense(units=label_units,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)

        self.permutation_encoding = permutation_encoding
        if permutation_encoding != 'no': 
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)


        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(ImbalancedMonet, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        #########################
        # Mixture concerns

        abs_permutation_mixture_logits = self.vector(None)
        abs_permutation_mixture = tf.nn.softmax(abs_permutation_mixture_logits)

        n_permutations = self.n_permutations
        if training:
            # Sample permutations that are not dropped out for each example
            # (B, DO)
            permutation_sample_do = sample_without_replacement(tf.zeros((batch_size, self.n_permutations)),
                                                               self.drop_out)

            # (B, DO, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_sample_do, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_sample_do, axis=0)

            # (B, DO)
            permutation_mixture_logits = tf.gather(abs_permutation_mixture_logits, permutation_sample_do, axis=0)
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits, axis=1) 
            n_permutations = self.drop_out
        else:
            # (P, T, T)
            permutation_matrices = self.permutation_matrices
            permutation_mixture_logits = abs_permutation_mixture_logits
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)


        perm_expend = tf.reshape(abs_permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])

        outputs_dict['mixture'] = abs_permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0) 
        outputs_dict['mixture_logits'] = tf.tile(tf.expand_dims(abs_permutation_mixture_logits,
                                                                axis=0),
                                                 (batch_size, 1))
       
        """
        for i in range(self.n_permutations):
            outputs_dict['mixture_logits_{}'.format(i)] = abs_permutation_mixture_logits[i]
        """

        #########################
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = n_permutations
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (B, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (B, P, T)
            y = tf.squeeze(tf.matmul(permutation_matrices, tf.expand_dims(y, axis=-1)), axis=-1)

            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, n_permutations, 1)), y], axis=-1)

        else:
            parallel = self.N_sample
            # (B, P)
            tiled_permutation_mixture_logits = tf.tile(tf.expand_dims(permutation_mixture_logits, axis=0),
                                                       (batch_size, 1))
            # (B, N)
            permutation_samples = tf.random.categorical(tiled_permutation_mixture_logits, self.N_sample)

            # (B, L, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_samples, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)


        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.permutation_encoding == 'perm_token':
            if training:
                kept_permutations = permutation_sample_do
            else:
                kept_permutations = permutation_samples

            # (P, P)
            permutation_onehots = tf.eye(self.n_permutations)

            # (B, k or L, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)



        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        # (B, ?, T + 1, T)
        padded_permutations = tf.concat([tf.zeros((batch_size, parallel, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=2)
        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, parallel))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)

            # (B, ?, 1)_
            y_k = tf.expand_dims(y_k, axis=-1)
            # (B, ?, T)
            y_k = tf.tile(y_k, multiples=[1, 1, self.n_task])

            # (B, ?, T)
            if k != 0:
                y_k = 2 * y_k - 1 

            # (1, ?, T)
            projection = padded_permutations[:, :, k, :]

            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, :, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, ?, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, ?)
                y_k = tf.squeeze(y_k, axis=-1)


        # (B, ?, T)
        logits = tf.concat(logits, axis=-1)
        # (B, ?, T, 1)
        logits = tf.expand_dims(logits, axis=-1)

        # (B, ?, T) 
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

        """
        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]
        """

        if training:

            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['prediction_logits'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (batch_size, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        pred = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        pred = tf.clip_by_value(pred, 0, 1)
        outputs_dict['global_pred'] = pred
        return outputs_dict

class GMonet(tkm.Model):
    def __init__(self,
                 G, # Number of groups
                 T, # Number of tasks by groups
                 P, # Number of permutations
                 k, # Dropout
                 recurrent_cell_args,
                 L, # Number of test samples
                 Lu=64,
                 **kwargs):
        super(GMonet, self).__init__()
        self.G = G
        self.T = T
        self.P = P
        self.L = L
        self.k = k
        self.Lu = Lu

        permutation_sampler = permutation_heuristics.sample_with_heuristic({"type": "random"})
        permutations = permutation_sampler(self.P, self.G)
        self.permutation_matrices = tf.Variable(np.identity(self.G)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (P, G)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        tf.print("permutation_mapping : ", self.permutation_mapping)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = self.G

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = PermutationDenseCategorical(n_classes=[self.T for i in range(self.G)])

        label_initializer = tf.keras.initializers.RandomNormal(mean=0.,
                                                               stddev=1./tf.math.sqrt(tf.dtypes.cast(self.T, tf.float32)))
        self.label_encoder = tkl.Dense(units=self.Lu,
                                       activation='linear',
                                       kernel_initializer=label_initializer)

        # Handling input compression
        self.input_compression = False
        self.U = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.U,
                                          activation='relu')

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.U)
        self.call(inputs=tf.ones((1, input_shape[-1])), mixture_logits=tf.zeros((self.P,)))


    def call(self, inputs, mixture_logits, training=None, y=None, **kwargs):
        outputs_dict = dict()
        B = tf.shape(inputs)[0]
        
        #########################
        # Mixture concerns
        # (P, )
        mixture = tf.nn.softmax(mixture_logits)

        if training:
            # Sample permutations that are not dropped out for each example
            # (B, k)
            parallel = self.k
            permutation_samples = sample_without_replacement(tf.zeros((B, self.P)),
                                                             self.k)
            # (B, k)
            sampled_mixture_logits = tf.gather(mixture_logits, permutation_samples, axis=0)
        else:
            # Sample permutation N permutations from mixture logits
            # (B, P)
            parallel = self.L
            tiled_permutation_mixture_logits = tf.tile(tf.expand_dims(mixture_logits, axis=0), (B, 1))
            # (B, N)
            permutation_samples = tf.random.categorical(tiled_permutation_mixture_logits, self.L)

            sampled_mixture_logits = tf.zeros((B, self.L))

        # (B, k or N)
        sampled_mixture = tf.nn.softmax(sampled_mixture_logits)
        # (B, k or L, G) 
        permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)

        perm_expend = tf.reshape(mixture, (self.P, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.G, self.G])
        outputs_dict['mixture'] = mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)
        
        outputs_dict['mixture_logits'] = tf.tile(tf.expand_dims(mixture_logits,
                                                                axis=0),
                                                 (B, 1))
       
        #########################
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1) 
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if training:
            # (B, 1, GT)
            y = tf.expand_dims(y, axis=1)
            # (B, k, GT)
            y = tf.tile(y, multiples=[1, parallel, 1])
            # (B, k, GT)
            y = 2 * y - 1
            # (B, k, GT)

        if self.input_compression:
            # (B, k or L, E)
            inputs = self.input_compresser(inputs, training=training)

        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = tf.zeros((B, parallel, self.G * self.T))

        y_k = tf.zeros((B, parallel, self.G * self.T))
        task_mask = tf.zeros((B, parallel, self.G * self.T))

        for k in range(self.G):
            if training:
                # (B, k, GT)
                y_k = y
            
            y_k = y_k * task_mask


            # (B, ?, Lu)
            y_k = self.label_encoder(y_k, training=training)


            # cell output : (B, k or L, I), states : (B, k or L, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, k or L, G * T)
            logit, task_mask = self.dense(permutation=permutation_mapping[:, :, k],
                                          inputs=cell_output,
                                          training=training)

            logits = logits + logit

            if not training:
                # (B, L, G * T)
                uniform_sampling = tf.random.uniform(shape=tf.shape(logit),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(logit) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, ?)
                y_k = 2 * y_k - 1

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = sampled_mixture_logits
            outputs_dict['loss']['mixture'] = sampled_mixture
            outputs_dict['loss']['prediction_logits'] = logits
        
        mixture = tf.tile(tf.reshape(sampled_mixture, (B, parallel, 1)), (1, 1, self.G * self.T))
        pred = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        pred = tf.clip_by_value(pred, 0, 1)
        outputs_dict['global_pred'] = pred
        return outputs_dict


SUPPORTED_MONET = {"monet": Monet,
                   "gmonet": GMonet,
                   "hard_monet": HardMonet,
                   "single_monet": SingleOrderMonet,
                   "imonet": ImbalancedMonet}
