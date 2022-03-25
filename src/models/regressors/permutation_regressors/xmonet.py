import models.regressors.permutation_regressors.permutation_heuristics as permutation_heuristics
import models.regressors.permutation_regressors.block_utils as block_utils
import models.regressors.permutation_regressors.utils as utils

from models.layers.recurrent_cells import recurrent_cell
from models.layers.permutation_layers import PermutationDense, PermutationDensev2, PermutationDenseCategorical, PermutationDenseCategoricalv2
from utils import sample_from_categorical, sample_without_replacement

import tensorflow.keras.layers as tkl
import tensorflow_probability as tfp
import tensorflow.keras.models as tkm
import tensorflow as tf
import numpy as np

import copy


class XMonet(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 dropout,
                 permutation_heuristic,
                 mixture_net,
                 recurrent_cell_args,
                 N_sample,
                 T=1.0,
                 **kwargs):
        super(XMonet, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        self.dropout = dropout
        self.T = T

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)

        # (M, T, T)
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
        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)

        # Handling input compression
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.mixture_net = mixture_net

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        #########################
        # Mixture concerns
        # (B, P)
        mixture_logits = self.mixture_net(inputs, training=training) / self.T
        abs_mixture_logits = mixture_logits

        # (B, P)
        abs_mixture = tf.reshape(tf.nn.softmax(abs_mixture_logits, axis=-1), (batch_size, self.n_permutations, 1, 1))
        # (B, P, T, T)
        abs_mixture = tf.tile(abs_mixture, (1, 1, self.n_task, self.n_task))
        # (B, P, T, T)
        abs_permutation_matrices = tf.tile(tf.expand_dims(self.permutation_matrices, axis=0),
                                           (batch_size, 1, 1, 1))
        # (B, T, T)
        soft_orders = tf.math.reduce_sum(abs_mixture * abs_permutation_matrices, axis=1)
        outputs_dict['soft_orders'] = soft_orders 

        if training:
            # DROPOUT:
            # Sample permutations that are not dropped out for each example.
            # (B, DO)
            kept_permutations = sample_without_replacement(tf.zeros((batch_size, self.n_permutations)),
                                                           self.dropout)
            n_permutations = self.dropout

        else:
            # INFERENCE:
            # Sample permutations based on the mixture_logits.
            # (B, N)
            kept_permutations = tf.random.categorical(mixture_logits,
                                                      self.N_sample,
                                                      dtype=tf.int32)
            n_permutations = self.N_sample


        # (B, P, T, T)
        permutation_matrices = tf.gather(self.permutation_matrices, kept_permutations, axis=0)
        permutation_mapping = tf.gather(self.permutation_mapping, kept_permutations, axis=0)

        # (B, P, 1)
        batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1), (1, n_permutations)), -1)
        # (B, P, 2)
        kept_permutations_index = tf.concat([batch_index, 
                                             tf.expand_dims(kept_permutations, axis=-1)],
                                            axis=-1)
        # (B, P)
        mixture_logits = tf.gather_nd(mixture_logits, kept_permutations_index)
        mixture = tf.nn.softmax(mixture_logits, axis=-1)

        #########################
        # Ground truth preparation for training
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)
            # (B, P, T)
            y = tf.tile(y, multiples=[1, n_permutations, 1])
            # (B, P, T)
            y = tf.squeeze(tf.matmul(permutation_matrices, tf.expand_dims(y, axis=-1)), axis=-1)
            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, n_permutations, 1)), y], axis=-1)

        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, P, I)
        inputs = tf.tile(inputs, multiples=[1, n_permutations, 1])

        # (B, P, E)
        inputs = self.input_compresser(inputs, training=training)
        outputs_dict['track_grad'] = inputs

        # (B, P, T + 1, T)
        padded_permutations = tf.concat([tf.zeros((batch_size, n_permutations, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=2)
        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, n_permutations))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)

            # (B, P, 1)_
            y_k = tf.expand_dims(y_k, axis=-1)

            # (B, P, T)
            y_k = tf.tile(2 * y_k - 1, multiples=[1, 1, self.n_task])

            # (B, P, T)
            projection = padded_permutations[:, :, k, :]
            y_k = y_k * projection

            # (B, P, 64)
            y_k = self.previous_label_encoder(y_k, training=training)


            # cell output : (B, P, I), states : (B, P, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, :, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, P, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                # (B, P, 1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, P)
                y_k = tf.squeeze(y_k, axis=-1)

        # (B, P, T)
        logits = tf.concat(logits, axis=-1)
        # (B, P, T, 1)
        logits = tf.expand_dims(logits, axis=-1)
        # (B, P, T) 
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = mixture_logits
            outputs_dict['loss']['mixture'] = mixture
            outputs_dict['loss']['output'] = logits

            mixture = tf.reshape(mixture, (batch_size, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
            outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)

        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
            outputs_dict['soft_orders'] = tf.math.reduce_mean(permutation_matrices, axis=1)
            outputs_dict['global_pred'] = tf.math.reduce_mean(tf.math.sigmoid(logits), axis=1)
        
        # (B, T)
        return outputs_dict

class XMonetExternalMixture(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 dropout,
                 permutation_heuristic,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding=False,
                 permutation_units=64,
                 **kwargs):
        super(XMonetExternalMixture, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        self.dropout = dropout
        self.permutation_encoding = permutation_encoding

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)

        # (M, T, T)
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

        if permutation_encoding:
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.0)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)

        # Handling input compression
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')


    def build(self, input_shape):
        self.call(inputs=tf.ones((1, input_shape[-1])), mixture_logits=tf.zeros((1, self.n_permutations)))
        # super(XMonetExternalMixture, self).build(input_shape)

    def call(self, inputs, mixture_logits, training=None, y=None, **kwargs):
        outputs_dict = dict()
        outputs_dict['track_grad'] = dict()
        batch_size = tf.shape(inputs)[0]

        #########################
        # Mixture concerns
        # (B, P)
        abs_mixture_logits = mixture_logits

        # (B, P)
        abs_mixture = tf.reshape(tf.nn.softmax(abs_mixture_logits, axis=-1), (batch_size, self.n_permutations, 1, 1))
        # (B, P, T, T)
        abs_mixture = tf.tile(abs_mixture, (1, 1, self.n_task, self.n_task))
        # (B, P, T, T)
        abs_permutation_matrices = tf.tile(tf.expand_dims(self.permutation_matrices, axis=0),
                                           (batch_size, 1, 1, 1))
        # (B, T, T)
        soft_orders = tf.math.reduce_sum(abs_mixture * abs_permutation_matrices, axis=1)
        outputs_dict['soft_orders'] = soft_orders 

        if training:
            # DROPOUT:
            # Sample permutations that are not dropped out for each example.
            # (B, DO)
            kept_permutations = sample_without_replacement(tf.zeros((batch_size, self.n_permutations)),
                                                           self.dropout)
            n_permutations = self.dropout

        else:
            # INFERENCE:
            # Sample permutations based on the mixture_logits.
            # (B, N)
            kept_permutations = tf.random.categorical(mixture_logits,
                                                      self.N_sample,
                                                      dtype=tf.int32)
            n_permutations = self.N_sample


        # (B, P, T, T)
        permutation_matrices = tf.gather(self.permutation_matrices, kept_permutations, axis=0)
        permutation_mapping = tf.gather(self.permutation_mapping, kept_permutations, axis=0)

        # (B, P, 1)
        batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1), (1, n_permutations)), -1)
        # (B, P, 2)
        kept_permutations_index = tf.concat([batch_index, 
                                             tf.expand_dims(kept_permutations, axis=-1)],
                                            axis=-1)
        # (B, P)
        mixture_logits = tf.gather_nd(mixture_logits, kept_permutations_index)
        mixture = tf.nn.softmax(mixture_logits, axis=-1)

        #########################
        # Ground truth preparation for training
        # (B, 1, I)
        if training:
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)
            # (B, P, T)
            y = tf.tile(y, multiples=[1, n_permutations, 1])
            # (B, P, T)
            y = tf.squeeze(tf.matmul(permutation_matrices, tf.expand_dims(y, axis=-1)), axis=-1)
            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, n_permutations, 1)), y], axis=-1)

        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, P, I)

        outputs_dict['track_grad']['xmonet_layer_0'] = inputs 
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, multiples=[1, n_permutations, 1])

        if self.permutation_encoding:
            # (P, P)
            permutation_onehots = tf.eye(self.n_permutations)
            # (B, k, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)

        inputs = self.input_compresser(inputs, training=training)
        # (B, P, T + 1, T)
        padded_permutations = tf.concat([tf.zeros((batch_size, n_permutations, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=2)
        #########################
        # Recurrent Loop
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, n_permutations))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)

            # (B, P, 1)_
            y_k = tf.expand_dims(y_k, axis=-1)

            # (B, P, T)
            y_k = tf.tile(2 * y_k - 1, multiples=[1, 1, self.n_task])

            # (B, P, T)
            projection = padded_permutations[:, :, k, :]
            y_k = y_k * projection

            # (B, P, 64)
            y_k = self.previous_label_encoder(y_k, training=training)


            # cell output : (B, P, I), states : (B, P, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, :, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, :, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, P, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                # (B, P, 1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, P)
                y_k = tf.squeeze(y_k, axis=-1)

        # (B, P, T)
        logits = tf.concat(logits, axis=-1)
        # (B, P, T, 1)
        logits = tf.expand_dims(logits, axis=-1)
        # (B, P, T) 
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = mixture_logits
            outputs_dict['loss']['mixture'] = mixture
            outputs_dict['loss']['output'] = logits

            mixture = tf.reshape(mixture, (batch_size, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
            outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)

        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
            outputs_dict['soft_orders'] = tf.math.reduce_mean(permutation_matrices, axis=1)
            outputs_dict['global_pred'] = tf.math.reduce_mean(tf.math.sigmoid(logits), axis=1)
        
        # (B, T)
        return outputs_dict

class PartialCategoricalXMonet(tkm.Model):
    def __init__(self,
                 C,
                 P,
                 k,
                 P_heuristic,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding=False,
                 permutation_units=64,
                 **kwargs):
        """
        C: list of classes cardinality for each tasks
        P: number of permutations
        k: number of ked permutations
        """
        super(PartialCategoricalXMonet, self).__init__()
        self.P = P
        self.T = len(C)
        self.C_tot = sum(C)
        self.N_sample = N_sample
        self.k = k
        self.permutation_encoding = permutation_encoding
        self.units = recurrent_cell_args['units']

        #########################
        # Task Mask:
        task_masks = []
        for i in range(self.T):
            task_mask_i = ([tf.zeros((C[j], )) for j in range(i)] +
                           [tf.ones((C[i], ))] + 
                           [tf.zeros((C[j], )) for j in range(i + 1, len(C))])
            task_masks.append(tf.expand_dims(tf.concat(task_mask_i, axis=0), axis=0))

        # (T + 1, C_tot)
        self.task_mask = tf.concat(task_masks, axis=0)

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(P_heuristic))
        permutations = permutation_sampler(self.P, self.T)

        # (P, T, T)
        self.permutation_matrices = tf.Variable(np.identity(self.T)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (P, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = self.T

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.prediction_dense1 = PermutationDensev2(n_task=self.T,
                                                    units=self.units,
                                                    activation='relu')
        self.prediction_dense2 = PermutationDenseCategoricalv2(n_classes=C,
                                                               activation='linear')


        label_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.label_encoder = tkl.Dense(units=label_units,
                                       activation='linear',
                                       kernel_initializer=label_encoder_initializer)
        if permutation_encoding:
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)

        # Handling input compression
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')


    def build(self, input_shape):
        self.call(inputs=tf.ones((1, input_shape[-1])), mixture_logits=tf.zeros((1, self.P)))
        # super(XMonetExternalMixture, self).build(input_shape)

    def call(self, inputs, mixture_logits, training=None, y=None, **kwargs):
        """
        inputs (B, I): input to the network
        mixture_logits (B, P): weighting logits for permutations
        """

        outputs_dict = dict()
        outputs_dict['track_grad'] = dict()
        B = tf.shape(inputs)[0]

        if training:
            # (B, T, C_tot)
            tiled_y = tf.tile(tf.expand_dims(y, axis=1), (1, self.T, 1))
            # (B, T, C_tot)
            tiled_task_mask = tf.tile(tf.expand_dims(self.task_mask, axis=0),
                                      (B, 1, 1))
            # (B, T)
            unsupervision_mask = tf.math.reduce_sum(tf.math.abs(tiled_task_mask * (tiled_y + 1)), axis=-1) == 0
            supervision_mask = 1 - tf.dtypes.cast(unsupervision_mask, dtype=tf.float32)
        else:
            supervision_mask = tf.ones((B, self.T))
        # tf.print("supervision_mask : ", supervision_mask)
                                                  
        #########################
        # Mixture concerns
        # (B, P)
        abs_mixture_logits = mixture_logits
        # (B, P)
        abs_mixture = tf.reshape(tf.nn.softmax(abs_mixture_logits, axis=-1), (B, self.P, 1, 1))
        # (B, P, T, T)
        abs_mixture = tf.tile(abs_mixture, (1, 1, self.T, self.T))
        # (B, P, T, T)
        abs_permutation_matrices = tf.tile(tf.expand_dims(self.permutation_matrices, axis=0),
                                           (B, 1, 1, 1))
        # (B, T, T)
        soft_orders = tf.math.reduce_sum(abs_mixture * abs_permutation_matrices, axis=1)
        outputs_dict['soft_orders'] = soft_orders 

        if training:
            # DROPOUT:
            # Sample permutations that are not dropped out for each example.
            # (B, k)
            kept_permutations = sample_without_replacement(tf.zeros((B, self.P)),
                                                           self.k)
            P = self.k

        else:
            # INFERENCE:
            # Sample permutations based on the mixture_logits.
            # (B, N)
            kept_permutations = tf.random.categorical(mixture_logits,
                                                      self.N_sample,
                                                      dtype=tf.int32)
            P = self.N_sample


        # (B, P, T, T)
        permutation_matrices = tf.gather(self.permutation_matrices, kept_permutations, axis=0)
        permutation_mapping = tf.gather(self.permutation_mapping, kept_permutations, axis=0)

        # (B, P, 1)
        batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(B), axis=-1), (1, P)), -1)
        # (B, P, 2)
        kept_permutations_index = tf.concat([batch_index, 
                                             tf.expand_dims(kept_permutations, axis=-1)],
                                            axis=-1)
        # (B, P)
        mixture_logits = tf.gather_nd(mixture_logits, kept_permutations_index)
        mixture = tf.nn.softmax(mixture_logits, axis=-1)

        #########################
        # Ground truth preparation for training
        # (B, 1, I)
        if training:
            # (B, 1, C_tot)
            y = tf.expand_dims(y, axis=1)
            # (B, P, C_tot)
            y = tf.tile(y, multiples=[1, P, 1])


        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        # (B, P, I)
        inputs = tf.tile(inputs, multiples=[1, P, 1])

        #########################
        # Permutation encoding
        if self.permutation_encoding:
            # (P, P)
            permutation_onehots = tf.eye(self.P)
            # (B, k or N, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)

        # (B, P, H)
        inputs = self.input_compresser(inputs, training=training)
        #########################
        # Initialisation
        # (B, P, H) 
        states = inputs
        # tf.print('states : ', states[0, 0, :])
        # (B, P, C_tot)
        logits_leq_k = tf.zeros((B, P, self.C_tot))
        # (B, P, C_tot)
        y_k = tf.zeros((B, P, self.C_tot))
        # (B, P)
        last_task_k = tf.zeros((B, P), dtype=tf.int32)
        #########################
        # Recurrent Loop
        # (B, ?, E)

        for k in range(self.T):
            if training:
                # (B, P, C_tot)
                last_task_mask_k = tf.gather(tf.concat([tf.zeros((1, self.C_tot)), self.task_mask], axis=0),
                                             last_task_k, axis=0)
                # tf.print("k: ", k)
                # tf.print("last_task_k: ", last_task_k[0, 0])
                # tf.print("last_task_mask_k", last_task_mask_k[0, 0, :])

                y_k = y * last_task_mask_k

            # (B, P, L_u)
            y_k = self.label_encoder(y_k, training=training)

            # (B, P)
            task_k = permutation_mapping[:, :, k]
            # tf.print('task_k : ', task_k[0, 0])

            # (B, P, 1)
            task_k_expanded = tf.expand_dims(task_k, axis=-1)
            # (B, P, 1)
            batch_range = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(B), axis=1), (1, P)), axis=-1)
            # (B, P)
            mask_k = tf.gather_nd(supervision_mask, indices=tf.concat([batch_range, task_k_expanded], axis=-1))

            # cell output : (B, P, I), states : (B, P, H)
            (cell_output, states) = self.recurrent_cell(permutation=task_k,
                                                        mask=mask_k,
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # tf.print('states : ', states[0, 0, :])
            # (B, P, C_tot)
            cell_output = self.prediction_dense1(inputs=cell_output,
                                                 permutation=task_k,
                                                 training=training)

            # (B, P, C_tot)
            logits_k = self.prediction_dense2(inputs=cell_output,
                                              permutation=task_k,
                                              training=training)

            # (B, P, C_tot)
            task_mask_k = tf.gather(self.task_mask, task_k)

            # (B, P, C_tot)
            logits_leq_k = logits_leq_k + logits_k
            # tf.print('logits_leq_k', logits_leq_k[0, 0, :])

            if not training: 
                # (B, N, C_tot)
                exp_logits_k = tf.math.exp(logits_k) * task_mask_k

                # (B, N, C_tot)
                p_k = exp_logits_k / tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_logits_k, axis=-1),
                                                            axis=-1),
                                             (1, 1, self.C_tot))

                # (B, N, C_tot)
                y_k = tfp.distributions.OneHotCategorical(probs=p_k, dtype=tf.float32).sample()

            # (B, P)
            int_mask_k = tf.dtypes.cast(mask_k, dtype=tf.int32)
            last_task_k = (1 + task_k) * int_mask_k + last_task_k * (1 - int_mask_k)
            # last_task_k = (1 + task_k) * mask_k + last_task_k * (1 - mask_k)

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = mixture_logits
            outputs_dict['loss']['mixture'] = mixture
            outputs_dict['loss']['prediction_logits'] = logits_leq_k
            outputs_dict['loss']['task_mask'] = self.task_mask
            outputs_dict['loss']['supervision_mask'] = supervision_mask

        # (B, P, T, C_tot)
        logits_leq_k = tf.tile(tf.expand_dims(logits_leq_k, axis=-2), (1, 1, self.T, 1))
        # (B, P, T, C_tot) 
        task_mask = tf.tile(tf.reshape(self.task_mask, (1, 1, self.T, self.C_tot)),
                            (B, P, 1, 1))
        # (B, P, T, C_tot)
        exp_logits_leq_k = tf.math.exp(logits_leq_k) * task_mask
        # (B, P, T, C_tot)
        pred_leq_k = exp_logits_leq_k / tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_logits_leq_k, axis=-1),
                                                               axis=-1),
                                                (1, 1, 1, self.C_tot))
        # (B, P, C_tot)
        pred_leq_k = tf.math.reduce_sum(pred_leq_k,
                                        axis=-2)

        if training:
            mixture = tf.reshape(mixture, (B, P, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.C_tot])
            # (B, C_tot)
            pred_leq_k = tf.math.reduce_sum(pred_leq_k * mixture, axis=1)
            # tf.print(pred_leq_k[0])
            outputs_dict['global_pred'] = pred_leq_k
        else:
            mixture = tf.ones((B, self.N_sample, self.T))/self.N_sample
            outputs_dict['soft_orders'] = tf.math.reduce_mean(permutation_matrices, axis=1)
            
            # (B, C_tot)
            pred_leq_k = tf.math.reduce_mean(pred_leq_k, axis=1)
            outputs_dict['global_pred'] = pred_leq_k 
        
        return outputs_dict

class CategoricalXMonet(tkm.Model):
    def __init__(self,
                 C,
                 P,
                 k,
                 P_heuristic,
                 recurrent_cell_args,
                 N_sample,
                 label_units=64,
                 permutation_encoding="no",
                 permutation_units=64,
                 **kwargs):
        """
        C: list of classes cardinality for each tasks
        P: number of permutations
        k: number of ked permutations
        """
        super(CategoricalXMonet, self).__init__()
        self.P = P
        self.T = len(C)
        self.C_tot = sum(C)
        self.N_sample = N_sample
        self.k = k
        self.permutation_encoding = permutation_encoding
        self.units = recurrent_cell_args['units']

        #########################
        # Task Mask:
        task_masks = []
        for i in range(self.T):
            task_mask_i = ([tf.zeros((C[j], )) for j in range(i)] +
                           [tf.ones((C[i], ))] + 
                           [tf.zeros((C[j], )) for j in range(i + 1, len(C))])
            task_masks.append(tf.expand_dims(tf.concat(task_mask_i, axis=0), axis=0))

        # (T + 1, C_tot)
        self.task_mask = tf.concat(task_masks, axis=0)

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(P_heuristic))
        permutations = permutation_sampler(self.P, self.T)

        # (P, T, T)
        self.permutation_matrices = tf.Variable(np.identity(self.T)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (P, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = self.T

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.prediction_dense1 = PermutationDensev2(n_task=self.T,
                                                    units=self.units,
                                                    activation='relu')
        self.prediction_dense2 = PermutationDenseCategoricalv2(n_classes=C,
                                                               activation='linear')


        label_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.label_encoder = tkl.Dense(units=label_units,
                                       activation='linear',
                                       kernel_initializer=label_encoder_initializer)
        if permutation_encoding != "no":
            permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
            self.permutation_encoder = tkl.Dense(units=permutation_units,
                                                 activation='linear',
                                                 kernel_initializer=permutation_encoder_initializer)

        # Handling input compression
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')


    def build(self, input_shape):
        self.call(inputs=tf.ones((1, input_shape[-1])), mixture_logits=tf.zeros((1, self.P)))
        # super(XMonetExternalMixture, self).build(input_shape)

    def call(self, inputs, mixture_logits, training=None, y=None, **kwargs):
        """
        inputs (B, I): input to the network
        mixture_logits (B, P): weighting logits for permutations
        """

        outputs_dict = dict()
        outputs_dict['track_grad'] = dict()
        B = tf.shape(inputs)[0]
                                                  
        #########################
        # Mixture concerns
        # (B, P)
        abs_mixture_logits = mixture_logits
        # (B, P)
        abs_mixture = tf.reshape(tf.nn.softmax(abs_mixture_logits, axis=-1), (B, self.P, 1, 1))
        # (B, P, T, T)
        abs_mixture = tf.tile(abs_mixture, (1, 1, self.T, self.T))
        # (B, P, T, T)
        abs_permutation_matrices = tf.tile(tf.expand_dims(self.permutation_matrices, axis=0),
                                           (B, 1, 1, 1))
        # (B, T, T)
        soft_orders = tf.math.reduce_sum(abs_mixture * abs_permutation_matrices, axis=1)
        outputs_dict['soft_orders'] = soft_orders 

        if training:
            # DROPOUT:
            # Sample permutations that are not dropped out for each example.
            # (B, k)
            kept_permutations = sample_without_replacement(tf.zeros((B, self.P)),
                                                           self.k)
            P = self.k

        else:
            # INFERENCE:
            # Sample permutations based on the mixture_logits.
            # (B, N)
            kept_permutations = tf.random.categorical(mixture_logits,
                                                      self.N_sample,
                                                      dtype=tf.int32)
            P = self.N_sample


        # (B, P, T, T)
        permutation_matrices = tf.gather(self.permutation_matrices, kept_permutations, axis=0)
        permutation_mapping = tf.gather(self.permutation_mapping, kept_permutations, axis=0)

        # (B, P, 1)
        batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(B), axis=-1), (1, P)), -1)
        # (B, P, 2)
        kept_permutations_index = tf.concat([batch_index, 
                                             tf.expand_dims(kept_permutations, axis=-1)],
                                            axis=-1)
        # (B, P)
        mixture_logits = tf.gather_nd(mixture_logits, kept_permutations_index)
        mixture = tf.nn.softmax(mixture_logits, axis=-1)

        #########################
        # Ground truth preparation for training
        # (B, 1, I)
        if training:
            # (B, 1, C_tot)
            y = tf.expand_dims(y, axis=1)
            # (B, P, C_tot)
            y = tf.tile(y, multiples=[1, P, 1])


        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        # (B, P, I)
        inputs = tf.tile(inputs, multiples=[1, P, 1])

        #########################
        # Permutation encoding
        if self.permutation_encoding == 'perm_token':
            # (P, P)
            permutation_onehots = tf.eye(self.P)
            # (B, k or N, P_u)
            permutation_encoding = self.permutation_encoder(tf.gather(permutation_onehots, kept_permutations, axis=0),
                                                            training=training)
            inputs = tf.concat([inputs, permutation_encoding], axis=-1)

        # (B, P, H)
        inputs = self.input_compresser(inputs, training=training)
        #########################
        # Initialisation
        # (B, P, H) 
        states = inputs
        # tf.print('states : ', states[0, 0, :])
        # (B, P, C_tot)
        logits_leq_k = tf.zeros((B, P, self.C_tot))
        # (B, P, C_tot)
        y_k = tf.zeros((B, P, self.C_tot))
        task_mask_k = tf.zeros((B, P, self.C_tot))

        if self.permutation_encoding == 'task_token':
            task_token = tf.zeros((B, P, self.T))
        #########################
        # Recurrent Loop
        # (B, ?, E)
        for k in range(self.T):
            if training:
                y_k = y * task_mask_k

            # (B, P, L_u)
            y_k = self.label_encoder(y_k, training=training)

            if (self.permutation_encoding == 'task_token') and (k!=0):
                # (B, P, T)
                task_token = task_token / tf.math.sqrt(k)
                # (B, P, P_u)
                task_encoding = self.permutation_encoder(task_token)
                y_k = tf.concat([y_k, task_encoding], axis=-1)

            # (B, P)
            task_k = permutation_mapping[:, :, k]

            # cell output : (B, P, I), states : (B, P, H)
            (cell_output, states) = self.recurrent_cell(permutation=task_k,
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, P, C_tot)
            cell_output = self.prediction_dense1(inputs=cell_output,
                                                 permutation=task_k,
                                                 training=training)

            # (B, P, C_tot)
            logits_k = self.prediction_dense2(inputs=cell_output,
                                              permutation=task_k,
                                              training=training)

            # (B, P, C_tot)
            task_mask_k = tf.gather(self.task_mask, task_k)


            if (self.permutation_encoding == 'task_token'):
                task_token = task_token + tf.gather(tf.eye(self.T), task_k, axis=0)

            # (B, P, C_tot)
            logits_leq_k = logits_leq_k + logits_k
            # tf.print('logits_leq_k', logits_leq_k[0, 0, :])

            if not training: 
                # (B, N, C_tot)
                exp_logits_k = tf.math.exp(logits_k) * task_mask_k

                # (B, N, C_tot)
                p_k = exp_logits_k / tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_logits_k, axis=-1),
                                                            axis=-1),
                                             (1, 1, self.C_tot))

                # (B, N, C_tot)
                y_k = tfp.distributions.OneHotCategorical(probs=p_k, dtype=tf.float32).sample()

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = mixture_logits
            outputs_dict['loss']['mixture'] = mixture
            outputs_dict['loss']['prediction_logits'] = logits_leq_k
            outputs_dict['loss']['task_mask'] = self.task_mask

        # (B, P, T, C_tot)
        logits_leq_k = tf.tile(tf.expand_dims(logits_leq_k, axis=-2), (1, 1, self.T, 1))
        # (B, P, T, C_tot) 
        task_mask = tf.tile(tf.reshape(self.task_mask, (1, 1, self.T, self.C_tot)),
                            (B, P, 1, 1))
        # (B, P, T, C_tot)
        exp_logits_leq_k = tf.math.exp(logits_leq_k) * task_mask
        # (B, P, T, C_tot)
        pred_leq_k = exp_logits_leq_k / tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_logits_leq_k, axis=-1),
                                                               axis=-1),
                                                (1, 1, 1, self.C_tot))
        # (B, P, C_tot)
        pred_leq_k = tf.math.reduce_sum(pred_leq_k,
                                        axis=-2)

        if training:
            mixture = tf.reshape(mixture, (B, P, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.C_tot])
            # (B, C_tot)
            pred_leq_k = tf.math.reduce_sum(pred_leq_k * mixture, axis=1)
            # tf.print(pred_leq_k[0])
            outputs_dict['global_pred'] = pred_leq_k
        else:
            mixture = tf.ones((B, self.N_sample, self.T))/self.N_sample
            outputs_dict['soft_orders'] = tf.math.reduce_mean(permutation_matrices, axis=1)
            
            # (B, C_tot)
            pred_leq_k = tf.math.reduce_mean(pred_leq_k, axis=1)
            outputs_dict['global_pred'] = pred_leq_k 
        
        return outputs_dict



SUPPORTED_XMONETS = {"xmonet": XMonet,
                     "xmonet_ext": XMonetExternalMixture,
                     "xmonet_pcategorical": PartialCategoricalXMonet,
                     "xmonet_categorical": CategoricalXMonet}
