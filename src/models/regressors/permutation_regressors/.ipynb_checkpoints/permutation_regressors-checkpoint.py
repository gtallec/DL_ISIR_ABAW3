import models.regressors.permutation_regressors.permutation_heuristics as permutation_heuristics
import models.regressors.permutation_regressors.block_utils as block_utils
import models.regressors.permutation_regressors.utils as utils

from models.layers.recurrent_cells import recurrent_cell
from models.layers.permutation_layers import PermutationDense, PermutationDensev2
from utils import sample_from_categorical, sample_without_replacement

import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm
import tensorflow as tf
import numpy as np

import copy


#################################
# One cell for all permutations #
#################################
class SoftOrderingRecurrentRegressor(tkm.Model):
    def __init__(self, n_task, n_permutations, permutation_heuristic, permutation_regressor, recurrent_cell_args, **kwargs):
        super(SoftOrderingRecurrentRegressor, self).__init__(**kwargs)
        self.n_task = n_task
        self.n_permutations = n_permutations
        #########################
        # Soft ordering matrix generation tools
        permutation_sampler = permutation_heuristics.sample_with_heuristic(permutation_heuristic)
        permutations = permutation_sampler(n_permutations, n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)
        self.permutation_regressor = permutation_regressor
        #########################
        # Recurrent Cell
        cell_type = recurrent_cell_args.pop('type')
        self.recurrent_cell = CELL_MAPPING[cell_type](**recurrent_cell_args)
        self.recurrent_denses = []
        for i in range(n_task):
            self.recurrent_denses.append(tkl.Dense(units=n_task,
                                                   activation='sigmoid'))

    def call(self, inputs, training=None, y=None, **kwargs):
        #########################
        # Generate permutation mixture coefficient
        initial_bias = (tf.ones((1, 4 * self.n_permutations))) 
        permutation_mixture = self.permutation_regressor(initial_bias)
        #########################

        permutation_mixture = tf.squeeze(permutation_mixture)
        permutation_mixture = tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1)
        permutation_mixture = tf.tile(permutation_mixture, 
                                      tf.constant([1, self.n_task, self.n_task]))

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
       
        T = self.n_task
        y_k = tf.zeros((tf.shape(inputs)[0], T))
        input_k = tf.concat([inputs, y_k], axis=1)
        state_k = [self.recurrent_cell.get_initial_state(inputs=input_k, dtype=tf.float32)]

        P_k = []
        outputs = dict()
        for k in range(1, T + 1):
            # One step of Main Recurrent Cell
            (cell_output, state_k) = self.recurrent_cell(inputs=input_k, states=state_k, training=training)
            y_pred_given_s_k = self.recurrent_denses[k-1](cell_output)

            P_k.append(y_pred_given_s_k)
            outputs['timestep_wise_good_branch_{}'.format(k)] = y_pred_given_s_k

            # Weighting y_pred_given_s by s proba i.e sigma
            diag_sigma_k = tf.linalg.diag(sigma[k-1, :])

            y_pred_k = tf.squeeze(tf.matmul(diag_sigma_k, tf.expand_dims(y_pred_given_s_k, -1))
                                  +
                                  tf.expand_dims((1 - sigma[k-1, :])/2, -1), axis=-1)
            outputs['timestep_wise_{}'.format(k)] = y_pred_k

            # If in training mode the next step input is ground truth for the task that has just been handled.
            if training:
                y_k = tf.squeeze(tf.matmul(diag_sigma_k, tf.expand_dims(y, -1))
                                 +
                                 tf.expand_dims((1 - sigma[k-1, :])/2, -1), axis=-1)

            # If in testing mode the next step input is last prediction.
            else:
                y_k = y_pred_k

            input_k = tf.concat([inputs, y_k], axis=1)

        P = tf.stack(P_k, axis=1)
        task_wise_output = tf.linalg.diag_part(tf.matmul(tf.transpose(sigma), P))
        outputs['task_wise'] = task_wise_output

        outputs['row_stochasticity'] = tf.math.reduce_sum(sigma, axis=0)
        outputs['column_stochasticity'] = tf.math.reduce_sum(sigma, axis=1)
        outputs['permutation_mixture'] = permutation_mixture
        return outputs

    def get_task_matrix(self):
        #########################
        # Generate permutation mixture coefficient
        initial_bias = (tf.ones((1, 4 * self.n_permutations))) 
        permutation_mixture = self.permutation_regressor(initial_bias)
        #########################

        permutation_mixture = tf.squeeze(permutation_mixture)
        permutation_mixture = tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1)
        permutation_mixture = tf.tile(permutation_mixture, 
                                      tf.constant([1, self.n_task, self.n_task]))

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
        return sigma

class OneCellPermutationNetwork(tkm.Model):
    def __init__(self, n_task, n_permutations, permutation_heuristic, vector, recurrent_cell_args, N_sample, permutation_encoding=False, ponderation='softmax', **kwargs):
        super(OneCellPermutationNetwork, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations],
                                                dtype=tf.float32,
                                                trainable=False)

        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = tkl.Dense(units=1, activation='linear')

        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='relu')

        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        # Handling permutation compression
        self.permutation_encoding = permutation_encoding
        self.permutation_encoder = tkl.Dense(units=2 * self.n_permutations,
                                             activation='relu')

        self.vector = vector
        self.ponderation = ponderation


    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(OneCellPermutationNetwork, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None, y=None, **kwargs):

        outputs_dict = dict()
        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)
        #########################
        # (batch_size, I)
        batch_size = tf.shape(inputs)[0]
        # (batch_size, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = self.n_permutations
            # (batch_size, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (batch_size, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (batch_size, P, T, 1)
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)

            # (batch_size, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
            permutation_samples = tf.range(0, self.n_permutations)
        else:
            parallel = self.N_sample
            permutation_samples = tf.squeeze(tf.random.categorical(tf.reshape(permutation_mixture_logits, (1, self.n_permutations)),
                                                                   num_samples=self.N_sample, dtype=tf.int32),
                                             axis=0)

        # (?, P)
        permutation_tokens = tf.gather(tf.eye(self.n_permutations), permutation_samples, axis=0)
        # (B, ?, 2 x P)
        if self.permutation_encoding:
            permutation_tokens = tf.tile(tf.expand_dims(self.permutation_encoder(permutation_tokens), axis=0), [batch_size, 1, 1])

        permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0) 

        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training) 

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))


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
            y_k = tf.tile(2 * y_k - 1, multiples=[1, 1, self.n_task])

            # (1, ?, T)
            projection = tf.reshape(tf.gather(padded_permutations[:, k, :],
                                              permutation_samples,
                                              axis=0), (1, parallel, self.n_task))

            # (B, ?, T)
            projection = tf.tile(projection, multiples=[batch_size, 1, 1])
            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)

            if self.permutation_encoding:
                y_k = tf.concat([y_k, permutation_tokens], axis=-1)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(inputs=y_k,
                                                        states=states,
                                                        training=training)
            print('states.shape : ', states[0].shape)
            # (B, ?, 1)
            cell_output = self.dense(inputs=cell_output, training=training)

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
        logits = tf.squeeze(tf.matmul(tf.gather(inv_permutation_matrices, permutation_samples, axis=0), logits), axis=-1)

        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]

        if training:

            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['output'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (1, self.n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[batch_size, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        return outputs_dict

################################################
# One Recurrent Cell / permutation Networks    #
################################################
class PermutationRNN(tkm.Model):
    def __init__(self, n_task, n_permutations, permutation_heuristic, recurrent_cell_args, l1ballvector, N_sample, label_encoding='no', **kwargs):
        super(PermutationRNN, self).__init__(**kwargs)
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        self.label_encoding = label_encoding
        #########################
        # Permutation Concerns
        permutation_sampler = permutation_heuristics.sample_with_heuristic(permutation_heuristic)
        permutations = permutation_sampler(n_permutations, n_task)

        # # permutation_matrices size : (n_permutations, n_task, n_task)
        self.permutations = tf.Variable(permutations, dtype=tf.float32, trainable=False)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)
        tf.print(self.permutation_matrices)
        # # inv_permutation_matrices size : (n_permutations, n_task, n_task)
        self.l1ballvector = l1ballvector
        #########################
        # Recurrent Cells
        self.recurrent_cells = []
        self.dense_by_permutation = []
        self.label_encoders = []
        if self.label_encoding == 'same':
            self.label_encoders.append(tkl.Dense(units=64,
                                                 activation='relu'))
        for i in range(n_permutations):
            if label_encoding == 'different':
                self.label_encoders.append(tkl.Dense(units=64,
                                                     activation='relu'))
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cells.append(recurrent_cell(cp_recurrent_cell_args))
            self.dense_by_permutation.append(tkl.Dense(units=1,
                                                       activation='linear'))
        #########################

    def call(self, inputs, training=None, y=None, **kwargs):
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))
        batchsize = tf.shape(inputs)[0]
        outputs_dict = dict()

        permutation_mixture = self.l1ballvector(None)
        perm_expend = tf.tile(tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1),
                              [1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices, axis=0)
        #########################
        # In training mode, generating ground truth for timesteps :
        if training:
            # tiled_y size (batch_size, n_perm, n_task=T, 1)
            tiled_y = tf.expand_dims(tf.tile(tf.expand_dims(y, 1),
                                             multiples=tf.constant([1, self.n_permutations, 1], dtype=tf.int64)),
                                     -1)
            # ground_truth (batch_size, n_perm, T)
            ground_truth = tf.squeeze(tf.matmul(self.permutation_matrices, tiled_y), axis=-1)
            #########################
            # Generate permutation mixture coefficient

            #########################
            # Input and State initialization
            list_inputs_ik = []
            list_states_ik = []

            for i in range(self.n_permutations):
                # Input initialization

                if self.label_encoding != 'no':
                    y_ik = tf.zeros((batchsize, self.n_task))
                    if self.label_encoding == 'same':
                        y_ik = self.label_encoders[0](y_ik, training=training)
                    else:
                        y_ik = self.label_encoders[i](y_ik, training=training)
                else:
                    y_ik = tf.zeros((batchsize, 1))

                input_ik = tf.concat([inputs, y_ik], axis=-1)
                list_inputs_ik.append(input_ik)
                
                # state_ik of size (batchsize, output_size)
                state_ik = self.recurrent_cells[i].get_initial_state(inputs=input_ik, dtype=tf.float32)
                list_states_ik.append(state_ik)
 
            #########################
            # Recurrent Loop
            list_outputs_ik = []
            for k in range(self.n_task):
                # # Cell output and state harvest
                list_outputs_i = []
                for i in range(self.n_permutations):
                    (cell_output_ik, state_ik) = self.recurrent_cells[i](inputs=list_inputs_ik[i],
                                                                         states=[list_states_ik[i]],
                                                                         training=training)


                    # # # output_i : (batch_size, 1)
                    output_ik = self.dense_by_permutation[i](cell_output_ik)

                    list_outputs_i.append(output_ik)

                    if self.label_encoding != 'no':
                        # (B, 1)
                        y_ik = 2 * tf.expand_dims(ground_truth[:, i, k], axis=-1) - 1

                        # (B, T)
                        y_ik = tf.tile(y_ik, multiples=[1, self.n_task])

                        # (B, T)
                        perm_tiled = tf.tile(tf.expand_dims(self.permutation_matrices[i, :, k], axis=0),
                                             multiples=[batchsize, 1])

                        # (B, T)
                        y_ik = y_ik * perm_tiled
                        
                        # (B, 64)
                        if self.label_encoding == 'same':
                            y_ik = self.label_encoders[0](y_ik, training=training)
                        else:
                            y_ik = self.label_encoders[i](y_ik, training=training)
                    else:
                        # (B, 1)
                        y_ik = tf.expand_dims(ground_truth[:, i, k], axis=1)
                    input_ik = tf.concat([inputs, y_ik], axis=-1)

                    list_inputs_ik[i] = input_ik
                    list_states_ik[i] = state_ik[0]
                list_outputs_ik.append(tf.expand_dims(tf.concat(list_outputs_i, axis=1),
                                                      axis=2))

            # outputs of size (batch_size, n_perm, T, 1)
            outputs = tf.expand_dims(tf.concat(list_outputs_ik, axis=-1), axis=-1)

            # outputs in order of size (batch_size, n_perm, T)
            outputs_in_order = tf.squeeze(tf.matmul(inv_permutation_matrices,
                                                    outputs),
                                          axis=-1)
            for i in range(self.n_task):
                key = 'timestep_wise_{}'.format(i)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = outputs_in_order[:, :, i]
                outputs_dict[key]['mixture'] = permutation_mixture
            for i in range(self.n_permutations):
                outputs_dict['permutation_wise_{}'.format(i)] = outputs_in_order[:, i, :]
            # print('outputs_in_order : ', outputs_in_order.shape)
            outputs_dict['permutation'] = dict()
            outputs_dict['permutation']['mixture'] = permutation_mixture
            outputs_dict['permutation']['output'] = outputs_in_order

        else:
            # Input and state initialization
            # simulation_inputs of size (batchsize, N, input_size)
            simulation_inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, self.N_sample, 1])
            list_inputs_ik = []
            list_states_ik = []

            for i in range(self.n_permutations):
                # Input initialization

                if self.label_encoding != 'no':
                    y_ik = tf.zeros((batchsize, self.N_sample, self.n_task))
                    if self.label_encoding == 'same':
                        y_ik = self.label_encoders[0](y_ik, training=training)
                    else:
                        y_ik = self.label_encoders[i](y_ik, training=training)
                else:
                    y_ik = tf.zeros((batchsize, self.N_sample, 1))
                input_ik = tf.concat([simulation_inputs, y_ik], axis=-1)

                list_inputs_ik.append(input_ik)
                
                # state_ik of size (batchsize, output_size)
                state_ik = self.recurrent_cells[i].get_initial_state(inputs=input_ik, dtype=tf.float32)
                # simulation_state_ik of size (batchsize, N_sample, output_size)
                simulation_state_ik = tf.tile(tf.expand_dims(state_ik, 1), multiples=[1, self.N_sample, 1])
                list_states_ik.append(simulation_state_ik)

            # Recurrent loop
            list_outputs_ik = []
            for k in range(self.n_task):
                list_outputs_i = []
                for i in range(self.n_permutations):
                    (cell_output_ik, state_ik) = self.recurrent_cells[i](inputs=list_inputs_ik[i],
                                                                         states=[list_states_ik[i]],
                                                                         training=training)
                    # output_i of size (batchsize, N, 1)
                    output_i_logits = self.dense_by_permutation[i](cell_output_ik, training=training)
                    output_i = tf.math.sigmoid(output_i_logits)
                    list_outputs_i.append(tf.expand_dims(output_i_logits, 2))
                    
                    # Sampling from output_i distrib : 
                    uniform_sampling = tf.random.uniform(shape=[batchsize, self.N_sample, 1],
                                                         minval=0,
                                                         maxval=1)
                    y_ik = tf.dtypes.cast(output_i - uniform_sampling > 0,
                                          dtype=tf.float32)

                    if self.label_encoding != 'no':
                        # (B, N, T)
                        y_ik = 2 * tf.tile(y_ik, multiples=[1, 1, self.n_task]) - 1
                        perm_tiled = tf.tile(tf.expand_dims(tf.expand_dims(self.permutation_matrices[i, :, k],
                                                                           axis=0),
                                                            axis=0),
                                             multiples=[batchsize, self.N_sample, 1])
                        y_ik = y_ik * perm_tiled
                        if self.label_encoding == 'same':
                            y_ik = self.label_encoders[0](y_ik, training=training)
                        else:
                            y_ik = self.label_encoders[i](y_ik, training=training)

                    input_ik = tf.concat([simulation_inputs, y_ik], axis=-1)

                    # State and input update : 
                    list_inputs_ik[i] = input_ik
                    list_states_ik[i] = state_ik[0]
                
                # Concatenation of output on permutation axis
                # outputs_k of size (batchsize, N, n_perm, 1)
                outputs_k = tf.concat(list_outputs_i, axis=2)
                list_outputs_ik.append(outputs_k)

            # size of permuted_outputs_ik (batchsize, N, n_perm, T, 1)
            permuted_outputs_ik = tf.expand_dims(tf.concat(list_outputs_ik, axis=-1), axis=-1)
            # size of outputs_ik (batchsize, N, n_perm, T)
            outputs_ik = tf.squeeze(tf.matmul(inv_permutation_matrices, permuted_outputs_ik), axis=-1)

            # size of mean_outputs_ik (batchsize, n_perm, T)
            mean_outputs_logits_ik = tf.math.reduce_mean(outputs_ik,
                                                         axis=1)
            mean_outputs_ik = tf.math.reduce_mean(tf.math.sigmoid(outputs_ik),
                                                  axis=1)

            for i in range(self.n_task):
                key = 'timestep_wise_{}'.format(i)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = mean_outputs_logits_ik[:, :, i]
                outputs_dict[key]['mixture'] = permutation_mixture

            for i in range(self.n_permutations):
                outputs_dict['permutation_wise_{}'.format(i)] = mean_outputs_logits_ik[:, i, :]

            outputs_dict['permutation'] = dict()
            outputs_dict['permutation']['mixture'] = permutation_mixture
            outputs_dict['permutation']['output'] = mean_outputs_logits_ik

            # Permutation mixture of size (1, n_perm):

            permutation_mixture = tf.expand_dims(permutation_mixture, axis=0)
            permutation_mixture = tf.expand_dims(permutation_mixture, axis=-1)
            permutation_mixture = tf.tile(permutation_mixture, multiples=[batchsize, 1, self.n_task])

            # E_sigma of size (batchsize, T)
            y_pred = tf.math.reduce_sum(mean_outputs_ik * permutation_mixture, axis=1)
            outputs_dict['task_wise'] = y_pred
        return outputs_dict

    def get_task_matrix(self):
        #########################
        # Generate permutation mixture coefficient
        permutation_mixture = self.l1ballvector(None)
        #########################

        permutation_mixture = tf.squeeze(permutation_mixture)
        permutation_mixture = tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1)
        permutation_mixture = tf.tile(permutation_mixture, 
                                      tf.constant([1, self.n_task, self.n_task]))

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
        return sigma.numpy()

    def get_optimal_order(self):
        permutation_mixture = self.l1ballvector(None)
        return self.permutations[tf.math.argmax(permutation_mixture)]

class SubPermutationRNN(tkm.Model):
    def __init__(self, specific_args, common_args, l1ballvectorlist, **kwargs):
        super(SubPermutationRNN, self).__init__(**kwargs)
        self.permutation_rnns = []
        vectors = l1ballvectorlist.instance.get_vectors()
        """
        tasks maps permutation rnn to the task they handle (eg : task[0] = [0, 1, 2], means that the first permutation rnn handles the three first tasks.
        """

        self.tasks = []
        sum_tasks = []
        for i in range(len(specific_args)):
            permutation_rnn = specific_args[i]
            tasks = permutation_rnn.pop('tasks')
            permutation_rnn['n_task'] = len(tasks)
            self.tasks.append(tasks)

            n_permutations = permutation_rnn['n_permutations']
            permutation_rnn['l1ballvector'] = vectors[i]

            self.permutation_rnns.append(PermutationRNN(**permutation_rnn, **copy.deepcopy(common_args)))
            sum_tasks = sum_tasks + tasks

        self.reverse_tasks = []
        for i in range(len(sum_tasks)):
            j = 0
            while sum_tasks[j] != i:
                j += 1
            self.reverse_tasks.append(j)



    def call(self, inputs, training=None, y=None, **kwargs):
        outputs = dict()
        if training:
            for i in range(len(self.permutation_rnns)):
                y_proj = tf.gather(y, self.tasks[i], axis=1)
                outputs['permutation_{}'.format(i)] = self.permutation_rnns[i](inputs=inputs, training=True, y=y_proj)['permutation']

        else:
            y_pred = []
            for i in range(len(self.permutation_rnns)):
                y_pred.append(self.permutation_rnns[i](inputs, training=False, y=None)['task_wise'])
            outputs['task_wise'] = tf.gather(tf.concat(y_pred, axis=1), self.reverse_tasks, axis=1)

        return outputs

    def get_task_matrices(self):
        task_matrices = []
        for i in range(len(self.permutation_rnns)):
            task_matrices.append(self.permutation_rnns[i].get_task_matrix())
        return task_matrices


class PermutationIm2Seq(tkm.Model):
    def __init__(self, n_task, n_permutations, vector, permutation_heuristic, recurrent_cell_args, N_sample, **kwargs):
        super(PermutationIm2Seq, self).__init__(**kwargs)
        self.N_sample = N_sample
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        #########################
        # Permutation Concerns
        permutation_sampler = permutation_heuristics.sample_with_heuristic(permutation_heuristic)
        permutations = permutation_sampler(n_permutations, n_task)

        # # permutation_matrices size : (n_permutations, n_task, n_task)
        self.permutations = tf.Variable(permutations, dtype=tf.float32, trainable=False)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)
        tf.print(self.permutation_matrices)
        self.vector = vector
        #########################
        # Recurrent Cells
        self.units = recurrent_cell_args.get('units')
        print("recurrent cell : ", recurrent_cell_args)
        self.recurrent_cells = []
        self.dense_by_permutation = []
        self.label_encoder = tkl.Dense(units=64,
                                       activation='linear')

        for i in range(n_permutations):
            self.recurrent_cells.append(recurrent_cell(copy.deepcopy(recurrent_cell_args)))
            self.dense_by_permutation.append(tkl.Dense(units=1,
                                                       activation='linear'))

        self.input_compression = False
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')
        #########################

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(PermutationIm2Seq, self).build(input_shape)

    def call(self, inputs, training=None, y=None):
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))
        batch_size = tf.shape(inputs)[0]
        outputs_dict = dict()

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)

        # (P, )
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)

        # (P, T, T)
        perm_expend = tf.tile(tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1),
                              [1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture

        # (T, T)
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices, axis=0)

        #########################
        # In training mode, generating ground truth for timesteps :
        if training:
            # tiled_y size (B, P, T, 1)
            tiled_y = tf.expand_dims(tf.tile(tf.expand_dims(y, 1),
                                             multiples=[1, self.n_permutations, 1]),
                                     -1)
            # ground_truth (B, P, T)
            ground_truth = tf.squeeze(tf.matmul(self.permutation_matrices, tiled_y), axis=-1)
            

            ################################
            # Input and State initialization
            list_inputs_i = []
            list_encoded_inputs_i = []
            list_states_i = []
            for i in range(self.n_permutations):
                list_inputs_i.append(tf.zeros((batch_size, self.n_task)))
                list_encoded_inputs_i.append(self.label_encoder(list_inputs_i[i]))
                if self.input_compression:
                    list_states_i.append(self.input_compresser(inputs))
                else:
                    list_states_i.append(inputs)

            #########################
            # Recurrent Loop
            list_logit_ki = []
            for k in range(self.n_task):
                list_logit_i = []
                for i in range(self.n_permutations):
                    # ((B, U), (B, U))
                    (cell_output_i, state_i) = self.recurrent_cells[i](inputs=list_encoded_inputs_i[i],
                                                                       states=[list_states_i[i]],
                                                                       training=training)
                    # (B, 1)
                    logit_i = self.dense_by_permutation[i](cell_output_i)
                    list_logit_i.append(logit_i)

                    # (B, T)
                    y_i = 2 * tf.tile(tf.expand_dims(ground_truth[:, i, k], axis=-1), multiples=[1, self.n_task]) - 1
                    list_inputs_i[i] = y_i * tf.tile(tf.expand_dims(self.permutation_matrices[i, k, :], axis=0), multiples=[batch_size, 1])
                    list_encoded_inputs_i[i] = self.label_encoder(list_inputs_i[i])

                    list_states_i[i] = state_i[0]
                
                # (B, P, 1)
                list_logit_ki.append(tf.expand_dims(tf.concat(list_logit_i, axis=1), axis=-1))

            # (B, P, T, 1)
            logits = tf.expand_dims(tf.concat(list_logit_ki, axis=-1), axis=-1)
            # (B, P, T)
            logits_in_order = tf.squeeze(tf.matmul(inv_permutation_matrices,
                                                   logits),
                                         axis=-1)

            for k in range(self.n_task):
                key = 'timestep_wise_{}'.format(k)
                outputs_dict[key] = dict()
                # (B, P)
                outputs_dict[key]['prediction'] = logits_in_order[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            for i in range(self.n_permutations):
                key = 'permutation_wise_{}'.format(i)
                # (B, T)
                outputs_dict[key] = logits_in_order[:, i, :]
            
            key = 'permutation'
            outputs_dict[key] = dict()
            outputs_dict[key]['mixture_logits'] = permutation_mixture_logits
            outputs_dict[key]['mixture'] = permutation_mixture
            outputs_dict[key]['output'] = logits_in_order

        else:

            ################################
            # Input and State initialization
            list_inputs_i = []
            list_encoded_inputs_i = []
            list_states_i = []

            for i in range(self.n_permutations):
                # (B, N, T)
                list_inputs_i.append(tf.zeros((batch_size, self.N_sample, self.n_task)))
                # (B, N, U)
                list_encoded_inputs_i.append(self.label_encoder(list_inputs_i[i]))
                # (B, N, U)

                if self.input_compression:
                    list_states_i.append(self.input_compresser(tf.tile(tf.expand_dims(inputs,
                                                                                      axis=1),
                                                                       multiples=[1, self.N_sample, 1])))
                else:
                    list_states_i.append(tf.tile(tf.expand_dims(inputs,
                                                                axis=1),
                                                 multiples=[1, self.N_sample, 1]))
            #########################
            # Recurrent Loop            
            list_logit_ki = []
            for k in range(self.n_task):
                list_logit_i = []
                for i in range(self.n_permutations):
                    # ((B, N, U), (B, N, U))
                    (cell_output_i, state_i) = self.recurrent_cells[i](inputs=list_encoded_inputs_i[i],
                                                                       states=[list_states_i[i]],
                                                                       training=training)
                    # (B, N, 1)
                    logit_i = self.dense_by_permutation[i](cell_output_i, training=training)
                    output_i = tf.sigmoid(logit_i)
                    list_logit_i.append(logit_i)

                    uniform_sampling = tf.random.uniform(shape=[batch_size, self.N_sample, 1],
                                                         minval=0,
                                                         maxval=1)
                    y_pred = tf.dtypes.cast(output_i - uniform_sampling > 0,
                                            dtype=tf.float32)

                    # (B, N, T)
                    y_tiled = 2 * tf.tile(y_pred, multiples=[1, 1, self.n_task]) - 1
                    # (B, N, T)
                    projection_vector = tf.tile(tf.expand_dims(tf.expand_dims(self.permutation_matrices[i, k, :],
                                                                              axis=0),
                                                               axis=0),
                                                multiples=[batch_size, self.N_sample, 1])

                    list_inputs_i[i] = y_tiled * projection_vector

                    # (B, N, U)
                    list_encoded_inputs_i[i] = self.label_encoder(list_inputs_i[i])
                    list_states_i[i] = state_i[0]

                # (B, N, P, 1)
                list_logit_ki.append(tf.expand_dims(tf.concat(list_logit_i, axis=-1), axis=-1))

            # (B, N, P, T, 1)
            logits = tf.expand_dims(tf.concat(list_logit_ki, axis=-1), axis=-1)

            # (B, N, P, T)
            logits_in_order = tf.squeeze(tf.matmul(inv_permutation_matrices, logits), axis=-1)

            # (B, P, T)
            mean_logits_in_order = tf.math.reduce_mean(logits_in_order,
                                                       axis=1)

            # (B, P, T)
            mean_outputs_in_order = tf.math.reduce_mean(tf.math.sigmoid(logits_in_order),
                                                        axis=1)

            for k in range(self.n_task):
                key = 'timestep_wise_{}'.format(k)
                outputs_dict[key] = dict()
                # (B, P)
                outputs_dict[key]['prediction'] = mean_logits_in_order[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            for i in range(self.n_permutations):
                key = 'permutation_wise_{}'.format(i)
                # (B, T)
                outputs_dict[key] = mean_logits_in_order[:, i, :]
            
            key = 'permutation'
            outputs_dict[key] = dict()
            outputs_dict[key]['mixture'] = permutation_mixture
            outputs_dict[key]['output'] = mean_logits_in_order
            
            permutation_mixture_expended = tf.tile(tf.expand_dims(tf.expand_dims(permutation_mixture, axis=0),
                                                                  axis=-1),
                                                   multiples=[batch_size, 1, self.n_task])
            output = tf.math.reduce_sum(mean_outputs_in_order * permutation_mixture_expended,
                                        axis=1)
            outputs_dict['task_wise'] = output
        return outputs_dict

    def get_optimal_order(self): 
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)
        return self.permutations[tf.math.argmax(permutation_mixture)]

class PermutationRNNv2(tkm.Model):
    def __init__(self, n_task, n_permutations, permutation_heuristic, recurrent_cell_args, vector, N_sample, previous_label_encoding='no', **kwargs):
        super(PermutationRNNv2, self).__init__(**kwargs)
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        #########################
        # Permutation Concerns
        permutation_sampler = permutation_heuristics.sample_with_heuristic(permutation_heuristic)
        permutations = permutation_sampler(n_permutations, n_task)

        # # permutation_matrices size : (n_permutations, n_task, n_task)
        self.permutations = tf.Variable(permutations, dtype=tf.float32, trainable=False)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)
        # # inv_permutation_matrices size : (n_permutations, n_task, n_task)
        self.vector = vector
        #########################
        # Recurrent Cells
        self.recurrent_cells = []
        self.dense_by_permutation = []
        self.previous_label_encoding = previous_label_encoding
        self.previous_label_encoders = []
        if self.previous_label_encoding == 'same':
            self.previous_label_encoders.append(tkl.Dense(units=64,
                                                          activation='relu'))
        for i in range(n_permutations):
            if self.previous_label_encoding == 'different':
                self.previous_label_encoders.append(tkl.Dense(units=64,
                                                              activation='relu'))
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cells.append(recurrent_cell(cp_recurrent_cell_args))
            self.dense_by_permutation.append(tkl.Dense(units=1,
                                                       activation='linear'))
        #########################

    def call(self, inputs, training=None, y=None, **kwargs):
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))
        batchsize = tf.shape(inputs)[0]
        outputs_dict = dict()

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)

        perm_expend = tf.tile(tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1),
                              [1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices, axis=0)
        #########################
        # In training mode, generating ground truth for timesteps :
        if training:
            # tiled_y size (batch_size, n_perm, n_task=T, 1)
            tiled_y = tf.expand_dims(tf.tile(tf.expand_dims(y, 1),
                                             multiples=tf.constant([1, self.n_permutations, 1], dtype=tf.int64)),
                                     -1)
            # ground_truth (batch_size, n_perm, T)
            ground_truth = tf.squeeze(tf.matmul(self.permutation_matrices, tiled_y), axis=-1)
            #########################
            # Generate permutation mixture coefficient

            #########################
            # Input and State initialization

            list_inputs_ik = []
            list_states_ik = []

            for i in range(self.n_permutations):
                # Input initialization

                if self.previous_label_encoding != 'no':
                    y_ik = tf.zeros((batchsize, self.n_task))
                    if self.previous_label_encoding == 'same':
                        y_ik = self.previous_label_encoders[0](y_ik, training=training)
                    else:
                        y_ik = self.previous_label_encoders[i](y_ik, training=training)
                else:
                    y_ik = tf.zeros((batchsize, 1))

                input_ik = tf.concat([inputs, y_ik], axis=-1)
                list_inputs_ik.append(input_ik)
                
                # state_ik of size (batchsize, output_size)
                state_ik = self.recurrent_cells[i].get_initial_state(inputs=input_ik, dtype=tf.float32)
                list_states_ik.append(state_ik)
 
            #########################
            # Recurrent Loop
            list_outputs_ik = []
            for k in range(self.n_task):
                # # Cell output and state harvest
                list_outputs_i = []
                for i in range(self.n_permutations):
                    (cell_output_ik, state_ik) = self.recurrent_cells[i](inputs=list_inputs_ik[i],
                                                                         states=[list_states_ik[i]],
                                                                         training=training)


                    # # # output_i : (batch_size, 1)
                    output_ik = self.dense_by_permutation[i](cell_output_ik)

                    list_outputs_i.append(output_ik)

                    if self.previous_label_encoding != 'no':
                        # (B, 1)
                        y_ik = 2 * tf.expand_dims(ground_truth[:, i, k], axis=-1) - 1

                        # (B, T)
                        y_ik = tf.tile(y_ik, multiples=[1, self.n_task])

                        # (B, T)
                        perm_tiled = tf.tile(tf.expand_dims(self.permutation_matrices[i, :, k], axis=0),
                                             multiples=[batchsize, 1])

                        # (B, T)
                        y_ik = y_ik * perm_tiled
                        
                        # (B, 64)
                        if self.previous_label_encoding == 'same':
                            y_ik = self.previous_label_encoders[0](y_ik, training=training)
                        else:
                            y_ik = self.previous_label_encoders[i](y_ik, training=training)
                    else:
                        # (B, 1)
                        y_ik = tf.expand_dims(ground_truth[:, i, k], axis=1)

                    input_ik = tf.concat([inputs, y_ik], axis=1)

                    list_inputs_ik[i] = input_ik
                    list_states_ik[i] = state_ik[0]
                list_outputs_ik.append(tf.expand_dims(tf.concat(list_outputs_i, axis=1),
                                                      axis=2))

            # outputs of size (batch_size, n_perm, T, 1)
            outputs = tf.expand_dims(tf.concat(list_outputs_ik, axis=-1), axis=-1)

            # outputs in order of size (batch_size, n_perm, T)
            outputs_in_order = tf.squeeze(tf.matmul(inv_permutation_matrices,
                                                    outputs),
                                          axis=-1)
            for i in range(self.n_task):
                key = 'timestep_wise_{}'.format(i)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = outputs_in_order[:, :, i]
                outputs_dict[key]['mixture'] = permutation_mixture
            for i in range(self.n_permutations):
                outputs_dict['permutation_wise_{}'.format(i)] = outputs_in_order[:, i, :]
            # print('outputs_in_order : ', outputs_in_order.shape)
            outputs_dict['permutation'] = dict()
            outputs_dict['permutation']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['permutation']['mixture'] = permutation_mixture
            outputs_dict['permutation']['output'] = outputs_in_order

        else:
            # Input and state initialization
            # simulation_inputs of size (batchsize, N, input_size)
            simulation_inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, self.N_sample, 1])
            list_inputs_ik = []
            list_states_ik = []

            for i in range(self.n_permutations):
                # Input initialization
                if self.previous_label_encoding != 'no':
                    y_ik = tf.zeros((batchsize, self.N_sample, self.n_task))
                    if self.previous_label_encoding == 'same':
                        y_ik = self.previous_label_encoders[0](y_ik, training=training)
                    else:
                        y_ik = self.previous_label_encoders[i](y_ik, training=training)
                else:
                    y_ik = tf.zeros((batchsize, self.N_sample, 1))
                input_ik = tf.concat([simulation_inputs, y_ik], axis=-1)
                list_inputs_ik.append(input_ik)
                
                # state_ik of size (batchsize, output_size)
                state_ik = self.recurrent_cells[i].get_initial_state(inputs=input_ik, dtype=tf.float32)
                # simulation_state_ik of size (batchsize, N_sample, output_size)
                simulation_state_ik = tf.tile(tf.expand_dims(state_ik, 1), multiples=[1, self.N_sample, 1])
                list_states_ik.append(simulation_state_ik)


            # Recurrent loop
            list_outputs_ik = []
            for k in range(self.n_task):
                list_outputs_i = []
                for i in range(self.n_permutations):
                    (cell_output_ik, state_ik) = self.recurrent_cells[i](inputs=list_inputs_ik[i],
                                                                         states=[list_states_ik[i]],
                                                                         training=training)
                    # output_i of size (batchsize, N, 1)
                    output_i_logits = self.dense_by_permutation[i](cell_output_ik, training=training)
                    output_i = tf.math.sigmoid(output_i_logits)
                    list_outputs_i.append(tf.expand_dims(output_i_logits, 2))
                    
                    # Sampling from output_i distrib : 
                    uniform_sampling = tf.random.uniform(shape=[batchsize, self.N_sample, 1],
                                                         minval=0,
                                                         maxval=1)
                    y_ik = tf.dtypes.cast(output_i - uniform_sampling > 0,
                                          dtype=tf.float32)

                    if self.previous_label_encoding != 'no':
                        # (B, N, T)
                        y_ik = 2 * tf.tile(y_ik, multiples=[1, 1, self.n_task]) - 1
                        perm_tiled = tf.tile(tf.expand_dims(tf.expand_dims(self.permutation_matrices[i, :, k],
                                                                           axis=0),
                                                            axis=0),
                                             multiples=[batchsize, self.N_sample, 1])
                        y_ik = y_ik * perm_tiled
                        if self.previous_label_encoding == 'same':
                            y_ik = self.previous_label_encoders[0](y_ik, training=training)
                        else:
                            y_ik = self.previous_label_encoders[i](y_ik, training=training)

                    input_ik = tf.concat([simulation_inputs, y_ik], axis=-1)

                    # State and input update : 
                    list_inputs_ik[i] = input_ik
                    list_states_ik[i] = state_ik[0]
                
                # Concatenation of output on permutation axis
                # outputs_k of size (batchsize, N, n_perm, 1)
                outputs_k = tf.concat(list_outputs_i, axis=2)
                list_outputs_ik.append(outputs_k)

            # size of permuted_outputs_ik (batchsize, N, n_perm, T, 1)
            permuted_outputs_ik = tf.expand_dims(tf.concat(list_outputs_ik, axis=-1), axis=-1)
            # size of outputs_ik (batchsize, N, n_perm, T)
            outputs_ik = tf.squeeze(tf.matmul(inv_permutation_matrices, permuted_outputs_ik), axis=-1)

            # size of mean_outputs_ik (batchsize, n_perm, T)
            mean_outputs_logits_ik = tf.math.reduce_mean(outputs_ik,
                                                         axis=1)
            mean_outputs_ik = tf.math.reduce_mean(tf.math.sigmoid(outputs_ik),
                                                  axis=1)

            for i in range(self.n_task):
                key = 'timestep_wise_{}'.format(i)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = mean_outputs_logits_ik[:, :, i]
                outputs_dict[key]['mixture'] = permutation_mixture

            for i in range(self.n_permutations):
                outputs_dict['permutation_wise_{}'.format(i)] = mean_outputs_logits_ik[:, i, :]

            outputs_dict['permutation'] = dict()
            outputs_dict['permutation']['mixture'] = permutation_mixture
            outputs_dict['permutation']['output'] = mean_outputs_logits_ik

            # Permutation mixture of size (1, n_perm):

            permutation_mixture = tf.expand_dims(permutation_mixture, axis=0)
            permutation_mixture = tf.expand_dims(permutation_mixture, axis=-1)
            permutation_mixture = tf.tile(permutation_mixture, multiples=[batchsize, 1, self.n_task])

            # E_sigma of size (batchsize, T)
            y_pred = tf.math.reduce_sum(mean_outputs_ik * permutation_mixture, axis=1)
            outputs_dict['task_wise'] = y_pred
        return outputs_dict

    def get_task_matrix(self):
        #########################
        # Generate permutation mixture coefficient
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)
        #########################

        permutation_mixture = tf.squeeze(permutation_mixture)
        permutation_mixture = tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1)
        permutation_mixture = tf.tile(permutation_mixture, 
                                      tf.constant([1, self.n_task, self.n_task]))

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
        return sigma.numpy()

    def get_optimal_order(self): 
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)
        return self.permutations[tf.math.argmax(permutation_mixture)]

class Pernetv3(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 permutation_heuristic,
                 recurrent_cell_args,
                 vector,
                 N_sample,
                 previous_label_encoding=False,
                 current_label_encoding=False,
                 **kwargs):

        super(Pernetv3, self).__init__(**kwargs)
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        #########################
        # Permutation Concerns
        permutation_sampler = permutation_heuristics.sample_with_heuristic(permutation_heuristic)
        permutations = permutation_sampler(n_permutations, n_task)

        # # permutation_matrices size : (n_permutations, n_task, n_task)
        self.permutations = tf.Variable(permutations, dtype=tf.float32, trainable=False)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)
        tf.print("permutation_matrices : ", self.permutation_matrices)
        self.vector = vector
        #########################
        # Recurrent Cells
        self.recurrent_cells = []
        self.dense_by_permutation = []
        self.previous_label_encoding = previous_label_encoding
        self.previous_label_encoder = tkl.Dense(units=32,
                                                activation='relu')
        
        self.current_label_encoding = current_label_encoding
        self.current_label_encoder = tkl.Dense(units=32,
                                               activation='relu')

        for i in range(n_permutations):
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cells.append(recurrent_cell(cp_recurrent_cell_args))
            self.dense_by_permutation.append(tkl.Dense(units=1,
                                                       activation='linear'))
        #########################

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]
        if training:
            parallel = []
            tiling = []
            y = tf.expand_dims(y, axis=1)
            y = tf.expand_dims(tf.tile(y, multiples=[1, self.n_permutations, 1]), axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
        else:
            parallel = [self.N_sample]
            tiling = [1]

        inputs = tf.expand_dims(inputs, axis=1)
        if not training:
            inputs = tf.expand_dims(inputs, axis=1)
        # (B, (N/()), P, I)
        inputs = tf.tile(inputs, multiples=[1, *parallel, self.n_permutations, 1])

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, self.n_task, 1)),
                                         self.permutation_matrices],
                                        axis=-1)
        y_k = tf.zeros((batch_size, *parallel, self.n_permutations))

        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))
        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)

        perm_expend = tf.tile(tf.expand_dims(tf.expand_dims(permutation_mixture, -1), -1),
                              [1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices, axis=0)
        #########################
        states_i = []
        logits_k = []
        for k in range(self.n_task):
            if training:
                # (batch_size, P)
                y_k = y[:, :, k]

            # (B, P, 1) or (B, N, P, 1)
            previous_label_encoded = tf.expand_dims(y_k, axis=-1)
            current_label_encoded = tf.zeros((batch_size, *parallel, self.n_permutations, 0))

            if self.previous_label_encoding:
                # (B, P, T) or (B, N, P, T)
                previous_label = tf.tile(2 * previous_label_encoded - 1, multiples=[1, *tiling, 1, self.n_task])
                # (1, P, T) or (1, 1, P, T)
                projection = tf.reshape(padded_permutations[:, :, k], (1, *tiling, self.n_permutations, self.n_task))
                # (B, P, T) or (B, N, P, T)
                projection = tf.tile(projection, multiples=[batch_size, *parallel, 1, 1])
                previous_label = previous_label * projection
                previous_label_encoded = self.previous_label_encoder(previous_label, training=training)

            if self.current_label_encoding:
                # (1, P, T) or (1, 1, P, T)
                current_label = tf.reshape(padded_permutations[:, :, k + 1], (1, *tiling, self.n_permutations, self.n_task))
                # (B, P, T) or (B, N, P, T)
                current_label = tf.tile(current_label, multiples=[batch_size, *parallel, 1, 1])
                current_label_encoded = self.current_label_encoder(current_label, training=training)
           
            input_k = tf.concat([inputs, previous_label_encoded, current_label_encoded], axis=-1)

            if k == 0:
                for i in range(self.n_permutations):
                    if training:
                        # (B, H)
                        input_i = input_k[:, i, :]
                    else:
                        # (B, 1, H)
                        input_i = input_k[:, :, i, :]
                    state_i_0 = self.recurrent_cells[i].get_initial_state(inputs=input_i,
                                                                          dtype=tf.float32)
                    if not training:
                        state_i_0 = tf.expand_dims(state_i_0, axis=1)

                    state_i_0 = tf.tile(state_i_0, multiples=[1, *parallel, 1])
                    states_i.append([state_i_0])

            logits_i = []
            ys_i = []

            for i in range(self.n_permutations):
                if training:
                    input_i = input_k[:, i, :]
                else:
                    input_i = input_k[:, :, i, :]

                (cell_output, state_i) = self.recurrent_cells[i](inputs=input_i,
                                                                 states=states_i[i],
                                                                 training=training)
                states_i[i] = state_i
                # (B, 1) or (B, N, 1)
                logit_i = self.dense_by_permutation[i](cell_output)
                logits_i.append(logit_i)

                if not training:
                    p_i = tf.math.sigmoid(logit_i)
                    uniform_sampling = tf.random.uniform(shape=[batch_size, self.N_sample, 1],
                                                         minval=0,
                                                         maxval=1)
                    # (B, N, 1)
                    y_i = tf.dtypes.cast(p_i - uniform_sampling > 0,
                                         dtype=tf.float32)
                    ys_i.append(y_i)

            if not training:
                # (B, N, P)
                y_k = tf.concat(ys_i, axis=-1)

            # (B, P, 1) or (B, N, P, 1)
            logit_k = tf.expand_dims(tf.concat(logits_i, axis=-1), axis=-1)
            logits_k.append(logit_k)

        # (B, P, T, 1) or (B, N, P, T, 1) 
        logits_reverted = tf.expand_dims(tf.concat(logits_k, axis=-1), axis=-1)
        # (B, P, T) or (B, N, P, T)
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits_reverted), axis=-1)
        outputs = tf.math.sigmoid(logits)

        if not training:
            # (B, P, T)
            logits = tf.math.reduce_mean(logits, axis=1)

            # (B, P, T)
            outputs = tf.math.reduce_mean(outputs, axis=1)

        for k in range(self.n_task):
            key = "timestep_wise_{}".format(k)
            outputs_dict[key] = dict()
            outputs_dict[key]['prediction'] = logits[:, :, k]
            outputs_dict[key]['mixture'] = permutation_mixture

        for i in range(self.n_permutations):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]

        outputs_dict['permutation'] = dict()
        outputs_dict['permutation']['mixture_logits'] = permutation_mixture_logits
        outputs_dict['permutation']['mixture'] = permutation_mixture
        outputs_dict['permutation']['output'] = logits

        permutation_mixture = tf.reshape(permutation_mixture, (1, self.n_permutations, 1))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[batch_size, 1, self.n_task])

        y_pred = tf.math.reduce_sum(outputs * permutation_mixture, axis=1)
        outputs_dict['global_pred'] = y_pred
        return outputs_dict

###################################################
# One Recurrent Cell / permutation of task blocks #
###################################################

class BlockPernet(tkm.Model):
    def __init__(self, 
                 blocks,
                 n_permutations,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 previous_label_encoding=False,
                 current_label_encoding=False,
                 **kwargs):
        super(BlockPernet, self).__init__(**kwargs)
        print('sub blocks : ', blocks)
        self.n_task = sum([len(block) for block in blocks])
        self.n_blocks = len(blocks)
        self.N_sample = N_sample
        self.n_permutations = n_permutations
        #########################
        # Permutation Concerns

        ###############
        # Input Permutation
        # From [0, 1, ..., n_task - 1] to [B1, B2, ...., BN]
        concat_blocks = block_utils.sum_block(blocks, [])
        self.to_block_order = tf.Variable(np.identity(self.n_task)[concat_blocks, :],
                                          dtype=tf.float32,
                                          trainable=False)
        tf.print('to_block_order :\n ', self.to_block_order)

        ###############
        # Block Permutation
        # From [B1, B2, ..., BN] to [Bs(1), Bs(2), ..., Bs(N)]

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        block_permutations = permutation_sampler(self.n_permutations, self.n_blocks)
        self.block_mapping = list(block_permutations)
        print(self.block_mapping)


        len_blocks = np.array([len(block) for block in blocks])

        self.block_permutations = tf.Variable(block_permutations, dtype=tf.float32, trainable=False)
        block_permutation_matrices = np.identity(self.n_blocks)[block_permutations]
        self.block_permutation_matrices = tf.Variable(block_permutation_matrices,
                                                      dtype=tf.float32,
                                                      trainable=False)
        permutation_matrices = []
        for i in range(self.n_permutations):
            permutation_matrices.append(block_utils.expand_to_blocks(block_permutation_matrices[i, :, :],
                                                                     len_blocks)[np.newaxis, :, :])

        self.len_blocks = tf.constant(len_blocks, dtype=tf.int32)
        self.permutation_matrices = tf.Variable(np.concatenate(permutation_matrices, axis=0),
                                                dtype=tf.float32,
                                                trainable=False)


        tf.print("block_permutations :\n ", self.block_permutations)
        tf.print("block permutation matrices :\n ", self.block_permutation_matrices)
        tf.print("permutation_matrices :\n ", self.permutation_matrices)

        #########################
        # Recurrent Cells
        self.recurrent_cells = []
        self.dense_by_block = []
        self.previous_label_encoding = previous_label_encoding
        self.previous_label_encoder = tkl.Dense(units=32,
                                                activation='relu')

        self.current_label_encoding = current_label_encoding
        self.current_label_encoder = tkl.Dense(units=32,
                                               activation='relu')
        for i in range(self.n_permutations):
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cells.append(recurrent_cell(cp_recurrent_cell_args))

        for i in range(self.n_blocks):
            model = tf.keras.Sequential()
            model.add(tkl.Dense(units=64,
                                activation='relu'))
            model.add(tkl.Dense(units=len(blocks[i]),
                                activation='linear'))
            self.dense_by_block.append(model)

        #########################
        # Mixture Concerns
        self.vector = vector

    def call(self, inputs, training=None, y=None, **kwargs):
        tiled_len_blocks = tf.tile(tf.reshape(self.len_blocks, (1, self.n_blocks, 1)),
                                   multiples=(self.n_permutations, 1, 1))
        tiled_len_blocks = tf.dtypes.cast(tiled_len_blocks, dtype=tf.dtypes.float32)
        tiled_len_blocks = tf.dtypes.cast(tf.squeeze(tf.matmul(self.block_permutation_matrices, tiled_len_blocks), axis=-1), dtype=tf.int32)

        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]
        if training:
            parallel = []
            tiling = []
            # (B, T, 1)
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.to_block_order, y), axis=-1)
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)
            # (B, P, T, 1)
            y = tf.expand_dims(tf.tile(y, multiples=[1, self.n_permutations, 1]), axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)
            # (B, P, T)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
        elif self.N_sample == 1:
            parallel = []
            tiling = []
        else:
            parallel = [self.N_sample]
            tiling = [1]

        if not training and self.N_sample != 1:
            inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, multiples=[1, *parallel, 1])

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        padded_len_blocks = tf.concat([tf.ones((self.n_permutations, 1), dtype=tf.dtypes.int32),
                                       tiled_len_blocks],
                                      axis=1)

        inv_to_block_order = tf.transpose(self.to_block_order, perm=(1, 0))
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)
        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_blocks, self.n_blocks])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.block_permutation_matrices,
                                                                axis=0)

        #########################
        # Recurrent Loop
        states_ki = []
        inputs_ki = []
        logits_ki = []
        ys_ki = [tf.zeros((batch_size, *parallel, 1)) for i in range(self.n_permutations)]
        block_cursor = tf.zeros((self.n_permutations, ), dtype=tf.dtypes.int32)

        for k in range(self.n_blocks):
            if training:
                for i in range(self.n_permutations):
                    # (batch_size, B_k)
                    ys_ki[i] = y[:, i, block_cursor[i]: block_cursor[i] + padded_len_blocks[i][k]]


            #########################
            # Previous Label encoding
            # List of P, (B, N, B_k) or (B, B_k)
            previous_label_encoded = ys_ki

            for i in range(self.n_permutations):
                # (B, N, B_k) or (B, B_k)
                previous_label = previous_label_encoded[i]
                # (B, N, B_k, 1) or (B, B_k, 1)
                previous_label = tf.expand_dims(previous_label, axis=-1) 
                # (B, N, B_k, T) or (B, B_k, T)
                previous_label = tf.tile(2 * previous_label - 1,
                                         multiples=[1, *tiling, 1, self.n_task])
                # (1, B_k, T) or (1, 1, B_k, T)
                projection = tf.reshape(padded_permutations[i, block_cursor[i]: block_cursor[i] + padded_len_blocks[i][k], :],
                                        (1, *tiling, padded_len_blocks[i][k], self.n_task))
                projection = tf.tile(projection, multiples=[batch_size, *parallel, 1, 1])
                # (B, B_k, T) or (B, N, B_k, T)
                previous_label = tf.math.reduce_sum(previous_label * projection, axis=-2)

                previous_label_encoded[i] = self.previous_label_encoder(previous_label,
                                                                        training=training)

                inputs_ki.append(tf.concat([inputs, previous_label_encoded[i]], axis=-1))

            if k == 0:
                for i in range(self.n_permutations):
                    input_0i = inputs_ki[i]
                    state_0i = self.recurrent_cells[i].get_initial_state(inputs=input_0i,
                                                                         dtype=tf.float32)
                    if not training and self.N_sample != 1:
                        state_0i = tf.expand_dims(state_0i, axis=1)

                    state_0i = tf.tile(state_0i, multiples=[1, *parallel, 1])
                    states_ki.append([state_0i])

            logits_i = []

            for i in range(self.n_permutations):
                input_ki = inputs_ki[i]
                (cell_output, state_ki) = self.recurrent_cells[i](inputs=input_ki,
                                                                  states=states_ki[i])
                states_ki[i] = state_ki
                block_k = self.block_mapping[i][k]
                logit_i = self.dense_by_block[block_k](cell_output)

                # (B, B_(k+1) or (B, N, B_(k+1))
                logits_i.append(logit_i)

                if not training:
                    # (B, N, B_(k+1))
                    p_i = tf.math.sigmoid(logit_i)
                    uniform_sampling = tf.random.uniform(shape=[batch_size,
                                                                *parallel,
                                                                padded_len_blocks[i][k + 1]],
                                                         minval=0,
                                                         maxval=1)
                    y_ki = tf.dtypes.cast(p_i - uniform_sampling > 0, dtype=tf.float32)
                    ys_ki[i] = y_ki
            logits_ki.append(logits_i)
            block_cursor = block_cursor + padded_len_blocks[:, k]


        logits_ik = [[logits_ki[k][i] for k in range(self.n_blocks)] for i in range(self.n_permutations)]
        for i in range(self.n_permutations):
            logits_i[i] = tf.expand_dims(tf.concat(logits_ik[i], axis=-1), axis=-2)

        logits_block_permutated = tf.expand_dims(tf.concat(logits_i, axis=-2), axis=-1)
        logits_block = tf.matmul(inv_permutation_matrices, logits_block_permutated)


        logits = tf.squeeze(tf.matmul(inv_to_block_order, logits_block), axis=-1)
        outputs = tf.math.sigmoid(logits)

        if not training and self.N_sample != 1:
            logits = tf.math.reduce_mean(logits, axis=1)
            outputs = tf.math.reduce_mean(outputs, axis=1)

        for k in range(self.n_task):
            key = "timestep_wise_{}".format(k)
            outputs_dict[key] = dict()
            outputs_dict[key]['prediction'] = logits[:, :, k]
            outputs_dict[key]['mixture'] = permutation_mixture

        for i in range(self.n_permutations):
            key = "permutation_wise_{}".format(i)
            outputs_dict[key] = logits[:, i, :]

        outputs_dict['permutation'] = dict()
        outputs_dict['permutation']['mixture_logits'] = permutation_mixture_logits
        outputs_dict['permutation']['mixture'] = permutation_mixture
        outputs_dict['permutation']['output'] = logits

        permutation_mixture = tf.reshape(permutation_mixture, (1, self.n_permutations, 1))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[batch_size, 1, self.n_task])

        y_pred = tf.math.reduce_sum(outputs * permutation_mixture, axis=1)
        outputs_dict['global_pred'] = y_pred
        return outputs_dict

class HierarchicalPernet(tkm.Model):
    def __init__(self, 
                 blocks,
                 n_permutations,
                 n_subpermutations,
                 permutation_heuristic,
                 vector,
                 subvector_list,
                 recurrent_cell_args,
                 N_sample,
                 **kwargs):
        super(HierarchicalPernet, self).__init__(**kwargs)
        print('blocks : ', blocks)
        self.blocks = blocks
        self.n_task = sum([len(block) for block in blocks])
        self.n_blocks = len(blocks)
        self.N_sample = N_sample
        self.n_permutations = n_permutations
        #########################
        # Permutation Concerns

        ###############
        # Input Permutation
        # From [0, 1, ..., n_task - 1] to [B1, B2, ...., BN]
        concat_blocks = block_utils.sum_block(blocks, [])
        self.to_block_order = tf.Variable(np.identity(self.n_task)[concat_blocks, :],
                                          dtype=tf.float32,
                                          trainable=False)
        tf.print('to_block_order :\n ', self.to_block_order)

        ###############
        # Block Permutation
        # From [B1, B2, ..., BN] to [Bs(1), Bs(2), ..., Bs(N)]

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        block_permutations = permutation_sampler(self.n_permutations, self.n_blocks)
        # coeff i, k is the k-th block treated by permutation i
        self.block_mapping = list(block_permutations)
        print('block mapping : ', self.block_mapping)


        len_blocks = np.array([len(block) for block in blocks])

        self.block_permutations = tf.Variable(block_permutations, dtype=tf.float32, trainable=False)
        block_permutation_matrices = np.identity(self.n_blocks)[block_permutations]
        self.block_permutation_matrices = tf.Variable(block_permutation_matrices,
                                                      dtype=tf.float32,
                                                      trainable=False)
        permutation_matrices = []
        for i in range(self.n_permutations):
            permutation_matrices.append(block_utils.expand_to_blocks(block_permutation_matrices[i, :, :],
                                                                     len_blocks)[np.newaxis, :, :])
        

        self.len_blocks = tf.constant(len_blocks, dtype=tf.int32)
        self.permutation_matrices = tf.Variable(np.concatenate(permutation_matrices, axis=0),
                                                dtype=tf.float32,
                                                trainable=False)


        tf.print("block_permutations :\n ", self.block_permutations) 
        tf.print("block permutation matrices :\n ", self.block_permutation_matrices)
        tf.print("permutation_matrices :\n ", self.permutation_matrices)

        #########################
        # Mixture Concerns
        self.vector = vector
 
        #########################
        # Recurrent Cells
        self.recurrent_cells = []
        self.model_by_block = []
        self.previous_label_encoder = tkl.Dense(units=32,
                                                activation='relu')

        self.current_label_encoder = tkl.Dense(units=32,
                                               activation='relu')
        for i in range(self.n_permutations):
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cells.append(recurrent_cell(cp_recurrent_cell_args))

        for i in range(self.n_blocks):
            model = SubBlockPernet(block_utils.relative_block_coords(blocks[i]),
                                   n_subpermutations[i],
                                   permutation_heuristic,
                                   subvector_list.instance.get_vector(i),
                                   recurrent_cell_args,
                                   previous_label_encoding=False,
                                   current_label_encoding=False,
                                   N_sample=N_sample,
                                   **kwargs)
            self.model_by_block.append(model)


    def call(self, inputs, training=None, y=None, **kwargs):
        tiled_len_blocks = tf.tile(tf.reshape(self.len_blocks, (1, self.n_blocks, 1)),
                                   multiples=(self.n_permutations, 1, 1))
        tiled_len_blocks = tf.dtypes.cast(tiled_len_blocks, dtype=tf.dtypes.float32)
        tiled_len_blocks = tf.dtypes.cast(tf.squeeze(tf.matmul(self.block_permutation_matrices, tiled_len_blocks), axis=-1), dtype=tf.int32)

        outputs_dict = dict()
        outputs_dict['loss'] = dict()

        for k in range(self.n_blocks):
            block_string = 'block_{}'.format(k)
            outputs_dict['loss'][block_string] = dict()
            outputs_dict['loss'][block_string]['block'] = block_utils.sum_block(self.blocks[k], [])
            outputs_dict['loss'][block_string]['mixtures'] = []
            outputs_dict['loss'][block_string]['mixtures_logits'] = []
            outputs_dict['loss'][block_string]['outputs'] = []

        batch_size = tf.shape(inputs)[0]
        if training:
            parallel = []
            tiling = []
            # (B, T, 1)
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.to_block_order, y), axis=-1)
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)
            # (B, P, T, 1)
            y = tf.expand_dims(tf.tile(y, multiples=[1, self.n_permutations, 1]), axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)
            # (B, P, T)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
        elif self.N_sample == 1:
            parallel = []
            tiling = []
        else:
            parallel = [self.N_sample]
            tiling = [1]

        if not training and self.N_sample != 1:
            inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, multiples=[1, *parallel, 1])

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        padded_len_blocks = tf.concat([tf.ones((self.n_permutations, 1), dtype=tf.dtypes.int32),
                                       tiled_len_blocks],
                                      axis=1)

        inv_to_block_order = tf.transpose(self.to_block_order, perm=(1, 0))
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)
        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_blocks, self.n_blocks])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.block_permutation_matrices,
                                                                axis=0)

        #########################
        # Recurrent Loop
        states_ki = []
        inputs_ki = []
        logits_ki = []
        ys_ki = [tf.zeros((batch_size, *parallel, 1)) for i in range(self.n_permutations)]
        block_cursor = tf.zeros((self.n_permutations, ), dtype=tf.dtypes.int32)
        # tf.print('block mapping : ', self.block_mapping)

        for k in range(self.n_blocks):
            if training:
                for i in range(self.n_permutations):
                    # (batch_size, B_k)
                    ys_ki[i] = y[:, i, block_cursor[i]: block_cursor[i] + padded_len_blocks[i][k]]
                    if k > 0:
                        block_at_time_k = self.block_mapping[i][k-1]
                        # print('block {} : '.format(block_at_time_k), ys_ki[i])


            #########################
            # Previous Label encoding
            # List of P, (B, N, B_k) or (B, B_k)
            previous_label_encoded = ys_ki

            for i in range(self.n_permutations):
                # (B, N, B_k) or (B, B_k)
                previous_label = previous_label_encoded[i]
                # (B, N, B_k, 1) or (B, B_k, 1)
                previous_label = tf.expand_dims(previous_label, axis=-1) 
                # (B, N, B_k, T) or (B, B_k, T)
                previous_label = tf.tile(2 * previous_label - 1,
                                         multiples=[1, *tiling, 1, self.n_task])
                # (1, B_k, T) or (1, 1, B_k, T)
                projection = tf.reshape(padded_permutations[i, block_cursor[i]: block_cursor[i] + padded_len_blocks[i][k], :],
                                        (1, *tiling, padded_len_blocks[i][k], self.n_task))
                projection = tf.tile(projection, multiples=[batch_size, *parallel, 1, 1])
                # (B, B_k, T) or (B, N, B_k, T)
                previous_label = tf.math.reduce_sum(previous_label * projection, axis=-2)

                previous_label_encoded[i] = self.previous_label_encoder(previous_label,
                                                                        training=training)

                inputs_ki.append(tf.concat([inputs, previous_label_encoded[i]], axis=-1))

            if k == 0:
                for i in range(self.n_permutations):
                    input_0i = inputs_ki[i]
                    state_0i = self.recurrent_cells[i].get_initial_state(inputs=input_0i,
                                                                         dtype=tf.float32)
                    if not training and self.N_sample != 1:
                        state_0i = tf.expand_dims(state_0i, axis=1)

                    state_0i = tf.tile(state_0i, multiples=[1, *parallel, 1])
                    states_ki.append([state_0i])

            logits_i = []

            for i in range(self.n_permutations):
                input_ki = inputs_ki[i]
                (cell_output, state_ki) = self.recurrent_cells[i](inputs=input_ki,
                                                                  states=states_ki[i])
                states_ki[i] = state_ki
                block_at_time_k = self.block_mapping[i][k]

                y_sup = None
                if training:
                    # y_sup is ground truth for the block at time k while y is ground truth for the block at time k-1
                    y_sup = y[:, i, block_cursor[i] + padded_len_blocks[i][k]: block_cursor[i] + padded_len_blocks[i][k] + padded_len_blocks[i][k+1]]
                    # print('block {} : '.format(block_at_time_k), y_sup)


                # print('cell_output.shape : ', cell_output.shape)     
                output_dict = self.model_by_block[block_at_time_k](cell_output,
                                                                   training=training,
                                                                   y=y_sup)
                block_string = 'block_{}'.format(block_at_time_k)

                outputs_dict['loss'][block_string]['mixtures'].append(tf.expand_dims(output_dict['permutation']['mixture'], axis=0))
                # (B, 1, P, B_{k+1})
                outputs_dict['loss'][block_string]['outputs'].append(tf.expand_dims(output_dict['permutation']['output'], axis=1))
                # (1, P)
                outputs_dict['loss'][block_string]['mixtures_logits'].append(tf.expand_dims(output_dict['permutation']['mixture_logits'], axis=0))

                outputs_dict['loss'][block_string]['block'] = block_utils.sum_block(self.blocks[block_at_time_k], [])
                outputs_dict['loss'][block_string]['permutation_matrix'] = output_dict['permutation_matrix']

                logit_i = output_dict['global_pred']
                # print("logit_i.shape : ", logit_i.shape)
                
                # (B, B_(k+1) or (B, N, B_(k+1))
                logits_i.append(logit_i)

                if not training:
                    # (B, N, B_(k+1))
                    p_i = tf.math.sigmoid(logit_i)
                    uniform_sampling = tf.random.uniform(shape=[batch_size,
                                                                *parallel,
                                                                padded_len_blocks[i][k + 1]],
                                                         minval=0,
                                                         maxval=1)
                    y_ki = tf.dtypes.cast(p_i - uniform_sampling > 0, dtype=tf.float32)
                    ys_ki[i] = y_ki
            logits_ki.append(logits_i)
            block_cursor = block_cursor + padded_len_blocks[:, k]


        logits_ik = [[logits_ki[k][i] for k in range(self.n_blocks)] for i in range(self.n_permutations)]
        for i in range(self.n_permutations):
            logits_i[i] = tf.expand_dims(tf.concat(logits_ik[i], axis=-1), axis=-2)
        
        logits_block_permutated = tf.expand_dims(tf.concat(logits_i, axis=-2), axis=-1)
        logits_block = tf.matmul(inv_permutation_matrices, logits_block_permutated)


        logits = tf.squeeze(tf.matmul(inv_to_block_order, logits_block), axis=-1)
        outputs = tf.math.sigmoid(logits)

        if not training and self.N_sample != 1:
            logits = tf.math.reduce_mean(logits, axis=1)
            outputs = tf.math.reduce_mean(outputs, axis=1)

        # print("logits.shape : ", logits.shape)
        blocks_permutation = []
        cursor = 0
        for k in range(self.n_blocks):
            block_string = 'block_{}'.format(k)
            block_size = self.len_blocks[k]
            top_padding = tf.zeros((cursor, block_size))
            down_padding = tf.zeros((self.n_task - cursor - block_size, block_size))
            block_permutation = outputs_dict['loss'][block_string]['permutation_matrix']
            outputs_dict['block{}_permutation_matrix'.format(k)] = block_permutation
            blocks_permutation.append(tf.concat([top_padding, block_permutation, down_padding], axis=0))
            cursor += block_size

        global_permutation_mixture = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        global_permutation_mixture = tf.tile(global_permutation_mixture, multiples=[1, self.n_task, self.n_task])
        global_permutation_matrix = tf.math.reduce_sum(global_permutation_mixture * self.permutation_matrices, axis=0)
        diag_block = tf.concat(blocks_permutation, axis=1)
        outputs_dict['global_permutation_matrix'] = tf.matmul(global_permutation_matrix, diag_block)

        for k in range(self.n_blocks):
            key = "blockwise_{}".format(k)
            outputs_dict[key] = dict()
            outputs_dict[key]['prediction'] = logits[:, :, k]
            outputs_dict[key]['mixture'] = permutation_mixture

        for i in range(self.n_permutations):
            key = "permutation_wise_{}".format(i)
            outputs_dict[key] = logits[:, i, :]

        outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
        outputs_dict['loss']['mixture'] = permutation_mixture
        outputs_dict['loss']['output'] = logits

        permutation_mixture = tf.reshape(permutation_mixture, (1, self.n_permutations, 1))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[batch_size, 1, self.n_task])

        y_pred = tf.math.reduce_sum(outputs * permutation_mixture, axis=1)
        outputs_dict['global_pred'] = y_pred
        return outputs_dict

class SubBlockPernet(tkm.Model):
    def __init__(self, 
                 blocks,
                 n_permutations,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 previous_label_encoding=False,
                 current_label_encoding=False,
                 **kwargs):
        super(SubBlockPernet, self).__init__(**kwargs)
        print('sub blocks : ', blocks)
        self.n_task = sum([len(block) for block in blocks])
        self.n_blocks = len(blocks)
        self.N_sample = N_sample
        self.n_permutations = n_permutations
        #########################
        # Permutation Concerns

        ###############
        # Input Permutation
        # From [0, 1, ..., n_task - 1] to [B1, B2, ...., BN]
        concat_blocks = block_utils.sum_block(blocks, [])
        self.to_block_order = tf.Variable(np.identity(self.n_task)[concat_blocks, :],
                                          dtype=tf.float32,
                                          trainable=False)
        tf.print('to_block_order :\n ', self.to_block_order)

        ###############
        # Block Permutation
        # From [B1, B2, ..., BN] to [Bs(1), Bs(2), ..., Bs(N)]

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        block_permutations = permutation_sampler(self.n_permutations, self.n_blocks)
        self.block_mapping = list(block_permutations)


        len_blocks = np.array([len(block) for block in blocks])

        self.block_permutations = tf.Variable(block_permutations, dtype=tf.float32, trainable=False)
        block_permutation_matrices = np.identity(self.n_blocks)[block_permutations]
        self.block_permutation_matrices = tf.Variable(block_permutation_matrices,
                                                      dtype=tf.float32,
                                                      trainable=False)
        permutation_matrices = []
        for i in range(self.n_permutations):
            permutation_matrices.append(block_utils.expand_to_blocks(block_permutation_matrices[i, :, :],
                                                                     len_blocks)[np.newaxis, :, :])
        

        self.len_blocks = tf.constant(len_blocks, dtype=tf.int32)
        self.permutation_matrices = tf.Variable(np.concatenate(permutation_matrices, axis=0),
                                                dtype=tf.float32,
                                                trainable=False)


        tf.print("block_permutations :\n ", self.block_permutations) 
        tf.print("block permutation matrices :\n ", self.block_permutation_matrices)
        tf.print("permutation_matrices :\n ", self.permutation_matrices)
 
        #########################
        # Recurrent Cells
        self.recurrent_cells = []
        self.dense_by_block = []
        self.previous_label_encoding = previous_label_encoding
        self.previous_label_encoder = tkl.Dense(units=32,
                                                activation='relu')

        self.current_label_encoding = current_label_encoding
        self.current_label_encoder = tkl.Dense(units=32,
                                               activation='relu')
        for i in range(self.n_permutations):
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cells.append(recurrent_cell(cp_recurrent_cell_args))

        for i in range(self.n_blocks):
            model = tf.keras.Sequential()
            model.add(tkl.Dense(units=64, 
                                activation='relu'))
            model.add(tkl.Dense(units=len(blocks[i]),
                                activation='linear'))
            self.dense_by_block.append(model)

        #########################
        # Mixture Concerns
        self.vector = vector
        print(self.vector.trainable_variables)

    def call(self, inputs, training=None, y=None, **kwargs):

        tiled_len_blocks = tf.tile(tf.reshape(self.len_blocks, (1, self.n_blocks, 1)),
                                   multiples=(self.n_permutations, 1, 1))
        tiled_len_blocks = tf.dtypes.cast(tiled_len_blocks, dtype=tf.dtypes.float32)
        tiled_len_blocks = tf.dtypes.cast(tf.squeeze(tf.matmul(self.block_permutation_matrices, tiled_len_blocks), axis=-1), dtype=tf.int32)

        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]
        if training:
            parallel = []
            tiling = []
            # (B, T, 1)
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.to_block_order, y), axis=-1)
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)
            # (B, P, T, 1)
            y = tf.expand_dims(tf.tile(y, multiples=[1, self.n_permutations, 1]), axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)
            # (B, P, T)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
        else:
            parallel = [self.N_sample]
            tiling = [1]

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        padded_len_blocks = tf.concat([tf.ones((self.n_permutations, 1), dtype=tf.dtypes.int32),
                                       tiled_len_blocks],
                                      axis=1)

        inv_to_block_order = tf.transpose(self.to_block_order, perm=(1, 0))
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        exp_permutation_mixture_logits = tf.math.exp(permutation_mixture_logits)
        permutation_mixture = exp_permutation_mixture_logits/tf.math.reduce_sum(exp_permutation_mixture_logits)
        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_blocks, self.n_blocks])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.block_permutation_matrices,
                                                                axis=0)

        #########################
        # Recurrent Loop
        states_ki = []
        inputs_ki = []
        logits_ki = []
        ys_ki = [tf.zeros((batch_size, *parallel, 1)) for i in range(self.n_permutations)]
        block_cursor = tf.zeros((self.n_permutations, ), dtype=tf.dtypes.int32)

        for k in range(self.n_blocks):
            if training:
                for i in range(self.n_permutations):
                    # (batch_size, B_k)
                    ys_ki[i] = y[:, i, block_cursor[i]: block_cursor[i] + padded_len_blocks[i][k]]


            #########################
            # Previous Label encoding
            # List of P, (B, N, B_k) or (B, B_k)
            previous_label_encoded = ys_ki
            # print("ys_ki.shape : ", ys_ki[0].shape)

            for i in range(self.n_permutations):
                # (B, N, B_k) or (B, B_k)
                previous_label = previous_label_encoded[i]
                # (B, N, B_k, 1) or (B, B_k, 1)
                previous_label = tf.expand_dims(previous_label, axis=-1) 
                # (B, N, B_k, T) or (B, B_k, T)
                previous_label = tf.tile(2 * previous_label - 1,
                                         multiples=[1, *tiling, 1, self.n_task])
                # (1, B_k, T) or (1, 1, B_k, T)
                projection = tf.reshape(padded_permutations[i, block_cursor[i]: block_cursor[i] + padded_len_blocks[i][k], :],
                                        (1, *tiling, padded_len_blocks[i][k], self.n_task))
                projection = tf.tile(projection, multiples=[batch_size, *parallel, 1, 1])
                # (B, B_k, T) or (B, N, B_k, T)
                previous_label = tf.math.reduce_sum(previous_label * projection, axis=-2)
                # print("previous_label : ", previous_label)

                previous_label_encoded[i] = self.previous_label_encoder(previous_label,
                                                                        training=training)

                inputs_ki.append(tf.concat([inputs, previous_label_encoded[i]], axis=-1))

            if k == 0:
                for i in range(self.n_permutations):
                    input_0i = inputs_ki[i]
                    state_0i = self.recurrent_cells[i].get_initial_state(inputs=input_0i,
                                                                         dtype=tf.float32)
                    if not training and self.N_sample != 1:
                        state_0i = tf.expand_dims(state_0i, axis=1)

                    state_0i = tf.tile(state_0i, multiples=[1, *parallel, 1])
                    states_ki.append([state_0i])

            logits_i = []

            for i in range(self.n_permutations):
                input_ki = inputs_ki[i]
                (cell_output, state_ki) = self.recurrent_cells[i](inputs=input_ki,
                                                                  states=states_ki[i])
                states_ki[i] = state_ki
                block_k = self.block_mapping[i][k]
                logit_i = self.dense_by_block[block_k](cell_output)
                
                # (B, B_(k+1) or (B, N, B_(k+1))
                logits_i.append(logit_i)

                if not training:
                    # (B, N, B_(k+1))
                    p_i = tf.math.sigmoid(logit_i)
                    uniform_sampling = tf.random.uniform(shape=[batch_size,
                                                                *parallel,
                                                                padded_len_blocks[i][k + 1]],
                                                         minval=0,
                                                         maxval=1)
                    y_ki = tf.dtypes.cast(p_i - uniform_sampling > 0, dtype=tf.float32)
                    ys_ki[i] = y_ki
            logits_ki.append(logits_i)
            block_cursor = block_cursor + padded_len_blocks[:, k]


        logits_ik = [[logits_ki[k][i] for k in range(self.n_blocks)] for i in range(self.n_permutations)]
        for i in range(self.n_permutations):
            logits_i[i] = tf.expand_dims(tf.concat(logits_ik[i], axis=-1), axis=-2)
        
        logits_block_permutated = tf.expand_dims(tf.concat(logits_i, axis=-2), axis=-1)
        logits_block = tf.matmul(inv_permutation_matrices, logits_block_permutated)


        logits = tf.squeeze(tf.matmul(inv_to_block_order, logits_block), axis=-1)
        # print('sub_blocks logits.shape : ', logits.shape)
        outputs = tf.math.sigmoid(logits)

        outputs_dict['permutation'] = dict()
        outputs_dict['permutation']['mixture_logits'] = permutation_mixture_logits
        outputs_dict['permutation']['mixture'] = permutation_mixture
        outputs_dict['permutation']['output'] = logits

        # (1, 1, P, 1) or (1, P, 1) 
        permutation_mixture = tf.reshape(permutation_mixture, (1, *tiling, self.n_permutations, 1))

        # (B, N, P, T) or (B, P, T) 
        permutation_mixture = tf.tile(permutation_mixture, multiples=[batch_size, *parallel, 1, self.n_task])

        if training:
            y_pred = tf.math.reduce_sum(outputs * permutation_mixture, axis=1)
        else:
            y_pred = tf.math.reduce_sum(outputs * permutation_mixture, axis=2)
        outputs_dict['global_pred'] = y_pred
        return outputs_dict

#########################################
# One Recurrent Cell per task Networks  #
#########################################
class ParallelizedPim2Seq(tkm.Model):
    def __init__(self, n_task, n_permutations, permutation_heuristic, vector, recurrent_cell_args, N_sample, ponderation='softmax', **kwargs):
        super(ParallelizedPim2Seq, self).__init__()
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
        self.dense = PermutationDense(n_task=n_task,
                                      units=1,
                                      activation='linear')
        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='relu')
        self.ponderation = ponderation

        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(ParallelizedPim2Seq, self).build(input_shape)


    @tf.function
    def call(self, inputs, training=None, y=None, **kwargs):
        # (batch_size, I)
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]
        print('batch_size : ', batch_size)
        # (batch_size, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        # (batch_size, 1, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = 1
            # (batch_size, 1, T)
            y = tf.expand_dims(y, axis=1)
            y = tf.tile(y, multiples=[1, self.n_permutations, 1])

            # (batch_size, P, T)
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)

            # (batch_size, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
        else:
            parallel = self.N_sample

        # (B, ?, P, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, self.n_permutations, 1])
        print(tf.shape(inputs))

        if self.input_compression:
            # (B, ?, P, E)
            inputs = self.input_compresser(inputs, training=training) 

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)

        #########################
        # Recurrent Loop
        # (B, ?, P, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, parallel, self.n_permutations))

        for k in range(self.n_task):
            if training:
                # (B, P)
                y_k = tf.gather(y, k, axis=-1)
                # (B, 1, P)
                y_k = tf.expand_dims(y_k, axis=-2)

            # (B, ?, P)
            previous_label = y_k
            # (B, ?, P, 1)_
            previous_label = tf.expand_dims(previous_label, axis=-1)

            # (B, ?, P, T)
            previous_label = tf.tile(2 * previous_label - 1, multiples=[1, 1, 1, self.n_task])

            # (1, 1, P, T)
            projection = tf.reshape(padded_permutations[:, k, :], (1, 1, self.n_permutations, self.n_task))

            # (B, ?, P, T)
            projection = tf.tile(projection, multiples=[batch_size, parallel, 1, 1])
            previous_label = previous_label * projection
            previous_label_encoded = self.previous_label_encoder(previous_label, training=training)


            # cell output : (B, ?, P, I)
            # states : (B, ?, P, H)
            (cell_output, states) = self.recurrent_cell(permutation=self.permutation_mapping[:, k],
                                                        inputs=previous_label_encoded,
                                                        states=states,
                                                        training=training)
            # (B, ?, P, 1)
            logit_k = self.dense(permutation=self.permutation_mapping[:, k],
                                 inputs=cell_output, training=training)

            logits.append(logit_k)


            if not training:
                # (B, ?, P, 1)
                p_k = tf.math.sigmoid(logit_k)
                uniform_sampling = tf.random.uniform(shape=tf.shape(logit_k),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(p_k - uniform_sampling > 0,
                                     dtype=tf.float32)
                y_k = tf.squeeze(y_k, axis=-1)


        # (B, ?, P, T)
        logits_permutated = tf.concat(logits, axis=-1)
        # (B, ?, P, T, 1)
        logits_permutated = tf.expand_dims(logits_permutated, axis=-1)

        # (B, ?, P, T) 
        logits_in_order = tf.squeeze(tf.matmul(inv_permutation_matrices, logits_permutated), axis=-1)
        outputs_in_order = tf.math.sigmoid(logits_in_order)

        logits_in_order = tf.math.reduce_mean(logits_in_order, axis=1)
        outputs_in_order = tf.math.reduce_mean(outputs_in_order, axis=1)


        for k in range(self.n_task):
            key = "timestep_wise_{}".format(k)
            outputs_dict[key] = dict()
            outputs_dict[key]['prediction'] = logits_in_order[:, :, k]
            outputs_dict[key]['mixture'] = permutation_mixture

        for i in range(self.n_permutations):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits_in_order[:, i, :]

        outputs_dict['loss'] = dict()
        outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
        outputs_dict['loss']['mixture'] = permutation_mixture
        outputs_dict['loss']['output'] = logits_in_order

        permutation_mixture = tf.reshape(permutation_mixture, (1, self.n_permutations, 1))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[batch_size, 1, self.n_task])

        y_pred = tf.math.reduce_sum(outputs_in_order * permutation_mixture, axis=1)
        outputs_dict['global_pred'] = y_pred
        return outputs_dict

class StochasticPim2Seq(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 permutation_encoding=None,
                 permutation_masking=False,
                 ponderation='softmax',
                 **kwargs):
        super(StochasticPim2Seq, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        self.permutation_masking = permutation_masking

        if permutation_masking:
            self.permutation_mask = tf.Variable(tf.ones((n_permutations, )), dtype=tf.float32, trainable=False)

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)

        # permutation mapping : (n_permutations, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = n_task

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = PermutationDense(n_task=n_task,
                                      units=1,
                                      activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)
        self.ponderation = ponderation

        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.permutation_encoding = permutation_encoding
        if permutation_encoding is not None:
            if permutation_encoding == "embedding":
                permutation_encoder_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
                self.permutation_encoder = tkl.Dense(units=self.n_permutations,
                                                     activation='linear',
                                                     kernel_initializer=permutation_encoder_initializer)

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(StochasticPim2Seq, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)

        n_permutations = self.n_permutations
        if self.permutation_masking and training:
            tf.print("permutation_mask : ", self.permutation_mask)
            permutation_matrices = tf.boolean_mask(self.permutation_matrices, self.permutation_mask, axis=0)
            permutation_mixture_logits = tf.boolean_mask(permutation_mixture_logits, self.permutation_mask)
            n_permutations = tf.dtypes.cast(tf.math.reduce_sum(self.permutation_mask), tf.int32)
        else:
            permutation_matrices = self.permutation_matrices

        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)

        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture


        perm_expend = tf.reshape(permutation_mixture, (n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * permutation_matrices,
                                                                axis=0)
        for i in range(self.n_permutations):
            outputs_dict['mixture_logits_{}'.format(i)] = permutation_mixture_logits[i]
        #########################
        batch_size = tf.shape(inputs)[0]
        # (B, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = n_permutations
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (B, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (B, P, T, 1)
            y = tf.expand_dims(y, axis=-1)

            rel_permutation_samples = tf.range(0, n_permutations)
            if self.permutation_masking:
                abs_permutation_samples = tf.boolean_mask(tf.range(0, self.n_permutations),
                                                          self.permutation_mask)
            else:
                abs_permutation_samples = rel_permutation_samples

            # (B, P, T)
            y = tf.squeeze(tf.matmul(permutation_matrices, y), axis=-1)

            # (B, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, n_permutations, 1)), y], axis=-1)

        else:
            parallel = self.N_sample
            # (1, P)
            tiled_permutation_mixture_logits = tf.expand_dims(permutation_mixture_logits, axis=0)
            rel_permutation_samples = tf.squeeze(tf.random.categorical(tiled_permutation_mixture_logits, self.N_sample), axis=0)
            abs_permutation_samples = rel_permutation_samples
            """
            if self.permutation_masking:
                abs_permutation_samples = tf.gather(tf.boolean_mask(tf.range(0, self.n_permutations), self.permutation_mask),
                                                    rel_permutation_samples, axis=0)
            else:
                abs_permutation_samples = rel_permutation_samples
            """


        if self.permutation_encoding is not None:
            # (?, P)
            permutation_tokens = tf.gather(tf.eye(self.n_permutations), abs_permutation_samples, axis=0)
            # (B, ?, 2 x P)

            if self.permutation_encoding == 'embedding':
                permutation_tokens = self.permutation_encoder(permutation_tokens)

            if self.permutation_encoding == 'n_token':
                P = tf.dtypes.cast(self.n_permutations, tf.float32)
                alpha = P / (tf.math.sqrt(P - 1) * (P - 2))
                beta = (1 - P) * alpha
                permutation_tokens = (beta - alpha) * permutation_tokens + alpha

            if self.permutation_encoding == 'symbole':
                # (P, )
                symboles = tf.range(start=0, limit=self.n_permutations, dtype=tf.float32)
                symboles = tf.tile(tf.expand_dims(symboles, axis=0), [parallel, 1])
                permutation_tokens = tf.math.reduce_sum(permutation_tokens * symboles, axis=1)
                permutation_tokens = tf.expand_dims(permutation_tokens, axis=1)

            permutation_tokens = tf.tile(tf.expand_dims(permutation_tokens, axis=0), [batch_size, 1, 1])
        # (P, T)
        permutation_mapping = tf.gather(self.permutation_mapping, abs_permutation_samples, axis=0)

        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        # (P, T + 1, T) 
        padded_permutations = tf.concat([tf.zeros((n_permutations, 1, self.n_task)),
                                         permutation_matrices],
                                        axis=1)
        # (P, T , T) 
        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 2, 1))
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
            y_k = tf.tile(2 * y_k - 1, multiples=[1, 1, self.n_task])

            # (1, ?, T)
            projection = tf.reshape(tf.gather(padded_permutations[:, k, :],
                                              rel_permutation_samples,
                                              axis=0), (1, parallel, self.n_task))

            # (B, ?, T)
            projection = tf.tile(projection, multiples=[batch_size, 1, 1])
            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)

            if self.permutation_encoding is not None:
                y_k = tf.concat([y_k, permutation_tokens], axis=-1)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, k],
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
        logits = tf.squeeze(tf.matmul(tf.gather(inv_permutation_matrices, rel_permutation_samples, axis=0), logits), axis=-1)

        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]

        if training:

            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['output'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (1, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[batch_size, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1) 
     

        return outputs_dict

    def change_mask(self, new_mask):
        self.permutation_mask.assign(new_mask)

    def get_task_matrix(self):
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        permutation_mixture = tf.reshape(permutation_mixture, tf.concat([tf.shape(permutation_mixture), [1, 1]],
                                                                        axis=0))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[1, self.n_task, self.n_task])

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
        return sigma.numpy()


class DropOutStochasticPim2Seq(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 drop_out,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 **kwargs):
        super(DropOutStochasticPim2Seq, self).__init__()
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
        self.dense = PermutationDensev2(n_task=n_task,
                                        units=1,
                                        activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='linear',
                                                kernel_initializer=previous_label_initializer)

        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(DropOutStochasticPim2Seq, self).build(input_shape)

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

            # (B, N, T, T)
            permutation_matrices = tf.gather(self.permutation_matrices, permutation_samples, axis=0)
            permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)


        inv_permutation_matrices = tf.transpose(permutation_matrices, perm=(0, 1, 3, 2))
        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

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
            y_k = tf.tile(2 * y_k - 1, multiples=[1, 1, self.n_task])

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
            """
            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture
            """

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['output'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (batch_size, n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[1, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        return outputs_dict

    def change_mask(self, new_mask):
        self.permutation_mask.assign(new_mask)

    def get_task_matrix(self):
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        permutation_mixture = tf.reshape(permutation_mixture, tf.concat([tf.shape(permutation_mixture), [1, 1]],
                                                                        axis=0))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[1, self.n_task, self.n_task])

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
        return sigma.numpy()

class T_StochasticPim2Seq(tkm.Model):
    def __init__(self,
                 n_task,
                 n_permutations,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 permutation_encoding=False,
                 ponderation='softmax',
                 **kwargs):
        super(T_StochasticPim2Seq, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)

        # Each task output distribution as its temperature :
        # (n_permutations , T) : Tij is the temperature for permutation i and task j
        self.log_T = tf.Variable(tf.zeros((n_permutations, n_task)), dtype=tf.float32, trainable=True)

        # permutation mapping : (n_permutations, T)
        self.permutation_mapping = tf.Variable(permutations, dtype=tf.int32, trainable=False)
        
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = n_task

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = PermutationDense(n_task=n_task,
                                      units=1,
                                      activation='linear')
        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='relu')
        self.ponderation = ponderation

        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')

        self.permutation_encoding = permutation_encoding
        if permutation_encoding:
            self.permutation_encoder = tkl.Dense(units=2 * self.n_permutations,
                                                 activation='relu')

        self.vector = vector

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(T_StochasticPim2Seq, self).build(input_shape)


    @tf.function
    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)
        #########################
        # (batch_size, I)
        batch_size = tf.shape(inputs)[0]
        # (batch_size, 1, I)
        inputs = tf.expand_dims(inputs, axis=1)
        if training:
            parallel = self.n_permutations
            # (batch_size, 1, T)
            y = tf.expand_dims(y, axis=1)

            # (batch_size, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])

            # (batch_size, P, T, 1)
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)

            # (batch_size, P, T + 1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
            permutation_samples = tf.range(0, self.n_permutations)
        else:
            parallel = self.N_sample
            permutation_samples = tf.squeeze(tf.random.categorical(tf.reshape(permutation_mixture_logits,
                                                                              (1, self.n_permutations)),
                                                                   num_samples=self.N_sample, dtype=tf.int32), axis=0)

        if self.permutation_encoding:
            # (?, P)
            permutation_tokens = tf.gather(tf.eye(self.n_permutations), permutation_samples, axis=0)
            # (B, ?, 2 x P)
            permutation_tokens = tf.tile(tf.expand_dims(self.permutation_encoder(permutation_tokens), axis=0), [batch_size, 1, 1])

        permutation_mapping = tf.gather(self.permutation_mapping, permutation_samples, axis=0)
        log_T = tf.gather(self.log_T, permutation_samples, axis=0) 

        # (B, ?, I)
        inputs = tf.tile(inputs, multiples=[1, parallel, 1])

        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))
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
            y_k = tf.tile(2 * y_k - 1, multiples=[1, 1, self.n_task])

            # (1, ?, T)
            projection = tf.reshape(tf.gather(padded_permutations[:, k, :],
                                              permutation_samples,
                                              axis=0), (1, parallel, self.n_task))

            # (B, ?, T)
            projection = tf.tile(projection, multiples=[batch_size, 1, 1])
            y_k = y_k * projection

            # (B, ?, 64)
            y_k = self.previous_label_encoder(y_k, training=training)

            if self.permutation_encoding:
                y_k = tf.concat([y_k, permutation_tokens], axis=-1)


            # cell output : (B, ?, I)
            # states : (B, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=permutation_mapping[:, k],
                                                        inputs=y_k,
                                                        states=states,
                                                        training=training)
            # (B, ?, 1)
            cell_output = self.dense(permutation=permutation_mapping[:, k],
                                     inputs=cell_output, training=training)

            logits.append(cell_output)

            if not training:
                # (B, ?, 1)
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                # (?, T)
                P = tf.shape(permutation_samples)[0] 
                mapping = tf.concat([tf.reshape(tf.range(0, P), (P, 1)),
                                     tf.reshape(permutation_mapping[:, k], (P, 1))], axis=1)
                # (1, ?, 1)
                log_T_k = tf.reshape(tf.gather_nd(log_T, mapping), (1, P, 1))

                # (B, ?, 1)
                T_k = tf.tile(tf.math.exp(log_T_k), multiples=(batch_size, 1, 1))

                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output/T_k) - uniform_sampling > 0,
                                     dtype=tf.float32)
                # (B, ?)
                y_k = tf.squeeze(y_k, axis=-1)


        # (B, ?, T)
        logits = tf.concat(logits, axis=-1)
        # (B, ?, T, 1)
        logits = tf.expand_dims(logits, axis=-1)

        # (B, ?, T) 
        logits = tf.squeeze(tf.matmul(tf.gather(inv_permutation_matrices, permutation_samples, axis=0), logits), axis=-1)



        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]

        if training:

            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['output'] = logits
            outputs_dict['loss']['log_T'] = self.log_T

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (1, self.n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[batch_size, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        T = tf.math.exp(log_T) 
        for i in range(self.n_task):
            outputs_dict["T_#{}".format(i)] = tf.math.exp(self.log_T)[:, i]

        outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits / T) * mixture, axis=1)
        return outputs_dict

    def get_task_matrix(self):
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        permutation_mixture = tf.reshape(permutation_mixture, tf.concat([tf.shape(permutation_mixture), [1, 1]],
                                                                        axis=0))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[1, self.n_task, self.n_task])

        sigma = tf.math.reduce_sum(permutation_mixture * self.permutation_matrices, axis=0)
        return sigma.numpy()


class APim2Seq(tkm.Model):
    def __init__(self, n_task, n_permutations, permutation_heuristic, vector, recurrent_cell_args, N_sample, ponderation='softmax', **kwargs):
        super(APim2Seq, self).__init__()
        self.n_task = n_task
        self.n_permutations = n_permutations
        self.N_sample = N_sample
        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        permutations = permutation_sampler(self.n_permutations, self.n_task)
        self.permutation_matrices = tf.Variable(np.identity(n_task)[permutations], dtype=tf.float32, trainable=False)
        self.permutation_mapping = list(permutations)
        tf.print(self.permutation_mapping)

        self.recurrent_cell_by_task = []
        self.dense_by_task = []

        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='relu')

        # Handling input compression
        self.input_compression = False
        self.units = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')


        for i in range(self.n_task):
            cp_recurrent_cell_args = copy.deepcopy(recurrent_cell_args)
            self.recurrent_cell_by_task.append(recurrent_cell(cp_recurrent_cell_args))
            self.dense_by_task.append(tkl.Dense(units=1, activation='linear'))
        self.vector = vector
        self.ponderation = ponderation


    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(APim2Seq, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()
        shape = tf.shape(inputs)
        batch_size = shape[0]
        if training:
            parallel = []
            tiling = []
            y = tf.expand_dims(y, axis=1)
            y = tf.expand_dims(tf.tile(y, multiples=[1, self.n_permutations, 1]), axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
        else:
            parallel = [self.N_sample]
            tiling = [1]
            inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, multiples=[1, *parallel, 1])

        if self.input_compression:
            inputs = self.input_compresser(inputs, training=training) 

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))

        #########################
        # Mixture concerns
        permutation_mixture_logits = self.vector(None)
        if self.ponderation == 'softmax':
            permutation_mixture = tf.nn.softmax(permutation_mixture_logits)
        elif self.ponderation == 'linear':
            permutation_mixture = tf.nn.relu(permutation_mixture_logits)
            sum_permutation_mixture = tf.math.reduce_sum(permutation_mixture)
            permutation_mixture = permutation_mixture / sum_permutation_mixture

        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_task, self.n_task])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.permutation_matrices,
                                                                axis=0)

        #########################
        # Recurrent Loop
        states_ki = [[tf.identity(inputs)] for i in range(self.n_permutations)]
        logits_ki = []
        ys_ki = [tf.zeros((batch_size, *parallel)) for i in range(self.n_permutations)]

        for k in range(self.n_task):
            logits_i = []
            for i in range(self.n_permutations):
                if training:
                    ys_ki[i] = y[:, i, k]

                # (B, N,)
                previous_label = ys_ki[i]
                # (B, N, 1)
                previous_label = tf.expand_dims(previous_label, axis=-1)

                # (B, N, T)
                previous_label = tf.tile(2 * previous_label - 1, multiples=[1, *tiling, self.n_task])

                projection = tf.reshape(padded_permutations[i, k, :], (1, *tiling, self.n_task))
                projection = tf.tile(projection, multiples=[batch_size, *parallel, 1])
                previous_label = previous_label * projection
                previous_label_encoded = self.previous_label_encoder(previous_label, training=training)

                task_at_time_k = self.permutation_mapping[i][k]

                (cell_output, state_i) = self.recurrent_cell_by_task[task_at_time_k](inputs=previous_label_encoded,
                                                                                     states=states_ki[i],
                                                                                     training=training)
                states_ki[i] = state_i
                # (B, N, 1)
                logit_i = self.dense_by_task[task_at_time_k](cell_output, training=training)
                logits_i.append(logit_i)

                if not training:
                    p_i = tf.math.sigmoid(logit_i)
                    uniform_sampling = tf.random.uniform(shape=[batch_size, self.N_sample, 1],
                                                         minval=0,
                                                         maxval=1)
                    y_i = tf.dtypes.cast(p_i - uniform_sampling > 0,
                                         dtype=tf.float32)
                    ys_ki[i] = tf.squeeze(y_i, axis=-1)

            # (?, P, 1)
            logit_k = tf.expand_dims(tf.concat(logits_i, axis=-1), axis=-1)
            logits_ki.append(logit_k)

        # (?, P, T, 1)
        logits_permutated = tf.expand_dims(tf.concat(logits_ki, axis=-1), axis=-1)

        # (?, P, T)
        logits = tf.squeeze(tf.matmul(inv_permutation_matrices, logits_permutated), axis=-1)
        outputs = tf.math.sigmoid(logits)

        if not training:
            logits = tf.math.reduce_mean(logits, axis=1)
            outputs = tf.math.reduce_mean(outputs, axis=1)


        for k in range(self.n_task):
            key = "timestep_wise_{}".format(k)
            outputs_dict[key] = dict()
            outputs_dict[key]['prediction'] = logits[:, :, k]
            outputs_dict[key]['mixture'] = permutation_mixture

        for i in range(self.n_permutations):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]

        outputs_dict['loss'] = dict()
        outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
        outputs_dict['loss']['mixture'] = permutation_mixture
        outputs_dict['loss']['output'] = logits

        permutation_mixture = tf.reshape(permutation_mixture, (1, self.n_permutations, 1))
        permutation_mixture = tf.tile(permutation_mixture, multiples=[batch_size, 1, self.n_task])

        y_pred = tf.math.reduce_sum(outputs * permutation_mixture, axis=1)
        outputs_dict['global_pred'] = y_pred
        return outputs_dict

class Maonet(tkm.Model):
    def __init__(self,
                 T,
                 recurrent_cell_args,
                 recurrent_mixture,
                 N_sample,
                 mixtX=True,
                 **kwargs):
        super(Maonet, self).__init__()
        self.T = T
        self.N_sample = N_sample
        self.mixtX = mixtX
  
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = T

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.recurrent_mixture = recurrent_mixture 

        self.dense = PermutationDensev2(n_task=T,
                                        units=1,
                                        activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.label_encoder = tkl.Dense(units=64,
                                       activation='linear',
                                       kernel_initializer=previous_label_initializer)
        self.E = 64

        # Handling input compression
        self.input_compression = False
        self.H = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.H,
                                          activation='relu')



    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.H)
        super(Maonet, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        """
        inputs: float (B, I).
        y: float (B, T).
        """
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        if self.input_compression:
            # (B, H)
            inputs = self.input_compresser(inputs, training=training)

        inputs = tf.expand_dims(inputs, axis=1)

        # (B, 1, H)
        state_mixture = inputs
        state_main = inputs

        if self.mixtX:
            state_mixture = inputs
        else:
            state_mixture = tf.zeros_like(inputs)

        mixtures = []
        logits_mixture = []
        logits_logsumexp = []
        logits_main = []
        permutations = []

        task_mapping = tf.range(self.T)
        T_factorial = tf.math.reduce_prod(task_mapping + 1)

        cur_leaf = 1
        y_k = tf.zeros((batch_size, cur_leaf, self.T))
        for k in range(self.T):
            #####
            # Mixture
            # encoded_input : (B, T!/(T - k)!, E)
            if training:
                mixture_mask = (y_k == 0)
                branches = self.T - k
                mult_to_leafs = tf.math.reduce_prod(tf.range(1, self.T - (k + 1) + 1))
                total_leafs = T_factorial
            else:
                mixture_mask = tf.dtypes.cast(y_k == 0, dtype=tf.float32)
                branches = self.N_sample
                mult_to_leafs = tf.math.pow(self.N_sample, self.T - (k + 1))
                total_leafs = tf.math.pow(self.N_sample, self.T)

            encoded_input = self.label_encoder(y_k)
            # output_mixture: (B, T!/(T - k)!, H), states_mixture: (B, T!/(T - k)!, H)
            (logit_mixture, state_mixture) = self.recurrent_mixture(inputs=encoded_input,
                                                                    states=[state_mixture],
                                                                    training=training)
            if training:
                # (B, T!/(T - k)!, T - k)
                logit_mixture = tf.boolean_mask(logit_mixture, mixture_mask)
                logit_mixture = tf.reshape(logit_mixture,
                                           (batch_size, cur_leaf, branches))

                mixture = tf.nn.softmax(logit_mixture, axis=-1)
                mixture = tf.reshape(mixture,
                                     (batch_size, cur_leaf * branches))

                # (B, T!/(T - k)!, T - k)
                logit_logsumexp = tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(logit_mixture, axis=-1), -1),
                                          (1, 1, branches))



                # (B, T!/(T - (k + 1))!)
                logit_mixture = tf.reshape(logit_mixture,
                                           (batch_size, cur_leaf * branches))
                # (B, 1, T!/(T - (k + 1))!)
                logit_mixture = tf.expand_dims(logit_mixture, axis=1)
                # (B, (T - (k + 1))!, T!/(T - (k + 1))!)
                logit_mixture = tf.tile(logit_mixture, (1, mult_to_leafs, 1))
                # (B, T!, 1)
                logit_mixture = tf.reshape(tf.transpose(logit_mixture, (0, 2, 1)),
                                           (batch_size, total_leafs, 1))
                logits_mixture.append(logit_mixture)


                # (B, T!/(T - (k + 1))!)
                logit_logsumexp = tf.reshape(logit_logsumexp,
                                             (batch_size, cur_leaf * branches))
                # (B, 1, T!/(T - (k + 1))!)
                logit_logsumexp = tf.expand_dims(logit_logsumexp, axis=1)
                # (B, (T - (k + 1))!, T!/(T - (k + 1))!)
                logit_logsumexp = tf.tile(logit_logsumexp, (1, mult_to_leafs, 1))
                # (B, T!, 1)
                logit_logsumexp = tf.reshape(tf.transpose(logit_logsumexp, (0, 2, 1)),
                                             (batch_size, total_leafs, 1))
                logits_logsumexp.append(logit_logsumexp)


                # Duplicate logit mixture:
                # logit_mixture : (B, 1, T!/(T-(k + 1))!)
                mixture = tf.expand_dims(mixture, axis=1)
                mixture = tf.tile(mixture, (1, mult_to_leafs, 1))

                # logit_mixture : (B, T!, 1)
                mixture = tf.reshape(tf.transpose(mixture, (0, 2, 1)),
                                     (batch_size, total_leafs, 1))
                mixtures.append(mixture)
            else:
                # (B, cur_leaf, T)
                masked_exp_logit = mixture_mask * tf.math.exp(logit_mixture)
                logit_mixture = tf.math.log(masked_exp_logit)
                # (B, cur_leaf, T)
                mixture = masked_exp_logit / tf.tile(tf.expand_dims(tf.math.reduce_sum(masked_exp_logit, axis=-1), axis=-1), (1, 1, self.T))
                # (B, T)
                mixture = tf.math.reduce_mean(mixture, axis=1)
                mixtures.append(tf.expand_dims(mixture, axis=-1))
 
            # Fancy Tiling for next timestep.
            # (B, 1, T!/(T - k)!, H)
            state_mixture = tf.expand_dims(state_mixture, axis=1)
            # (B, T - k, T!/(T - k)!, H)
            state_mixture = tf.tile(state_mixture, (1, branches, 1, 1))
            # (B, T!/(T - (k + 1))!, H)
            state_mixture = tf.reshape(tf.transpose(state_mixture, (0, 2, 1, 3)),
                                       (batch_size, cur_leaf * branches, self.H))


            #####
            # Main Network
            # Fancy tiling main input
            # (B, 1, T!/(T - k)!, E)
            encoded_input = tf.expand_dims(encoded_input, axis=1)
            # (B, T - k, T! / (T - k)!, E)
            encoded_input = tf.tile(encoded_input, multiples=(1, branches, 1, 1))
            # (B, T!/(T - (k + 1))!, E)
            encoded_input = tf.reshape(tf.transpose(encoded_input, (0, 2, 1, 3)),
                                       (batch_size, cur_leaf * branches, self.E))


            # Fancy tiling main state
            # (B, 1, T!/(T - k)!, H)
            state_main = tf.expand_dims(state_main, axis=1)
            # (B, T - k, T!/(T - k)!, H)
            state_main = tf.tile(state_main, (1, branches, 1, 1))
            # (B, T!/(T - (k + 1))!, H)
            state_main = tf.reshape(tf.transpose(state_main, (0, 2, 1, 3)),
                                    (batch_size, cur_leaf * branches, self.H))


            if training:
                # (cur_leaf * branches,)
                ravel_tm = tf.reshape(task_mapping, (cur_leaf * branches,))
                # (B, cur_leaf * branches)
                permutation = tf.tile(tf.expand_dims(ravel_tm, axis=0),
                                      multiples=(batch_size, 1))
                # Fancy tiling
                # (1, cur_leaf * branches)
                permutation_tm = tf.expand_dims(ravel_tm, axis=0)
                # (total_leafs, 1)
                permutation_tm = tf.reshape(tf.transpose(tf.tile(permutation_tm, (mult_to_leafs, 1)), (1, 0)),
                                            (total_leafs, 1))
                # (B, total_leafs, 1)
                permutation_tm = tf.tile(tf.expand_dims(permutation_tm, axis=0), (batch_size, 1, 1))

            else:
                # (B * cur_leaf, T)
                logit_mixture = tf.reshape(logit_mixture, (batch_size * cur_leaf, self.T))
                # (B * cur_leaf, branches)
                permutation = tf.random.categorical(logit_mixture, branches)
                # (B, cur_leaf * branches)
                permutation = tf.reshape(permutation,
                                         (batch_size, cur_leaf * branches))
                permutation_tm = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(permutation, axis=1), (1, mult_to_leafs, 1)),
                                                         (0, 2, 1)),
                                            (batch_size, total_leafs, 1))

            permutations.append(permutation_tm)


            # (B, cur_leaf * branches, H)
            output_main, state_main = self.recurrent_cell(inputs=encoded_input,
                                                          states=state_main,
                                                          permutation=permutation,
                                                          training=training)

            # (B, cur_leaf * branches, 1)
            logit_main = self.dense(output_main,
                                    permutation=permutation,
                                    training=training)

            pred_main = tf.math.sigmoid(logit_main)

            # Duplicate logit_main :
            # logit_main : (B, 1, cur_leaf * branches, 1)
            logit_main = tf.expand_dims(logit_main, axis=1)
            # logit_main : (B, mult_to_leafs, cur_leaf * branches, 1)
            logit_main = tf.tile(logit_main, (1, mult_to_leafs, 1, 1))
            # logit_main : (B, total_leafs, 1)
            logit_main = tf.reshape(tf.transpose(logit_main, (0, 2, 1, 3)),
                                    (batch_size, total_leafs, 1))
                
            logits_main.append(logit_main)


            # Fancy tiling ground truth:
            # (B, 1, T! / (T - k)!, T)
            y_k = tf.expand_dims(y_k, axis=1)
            # (B, T - k, T!/(T-k)!, T) 
            y_k = tf.tile(y_k, (1, branches, 1, 1))
            y_k = tf.reshape(tf.transpose(y_k, (0, 2, 1, 3)),
                             (batch_size, cur_leaf * branches, self.T))


            if training:
                # (B, T!/(T - (k + 1))!)
                y_k_new = 2 * tf.gather(y, ravel_tm, axis=1) - 1
                # (B, T!/(T - (k + 1))!, T)
                y_k_new = tf.tile(tf.expand_dims(y_k_new, axis=-1), (1, 1, self.T))

                # (B, T!/(T - (k + 1))!, T)
                task_mapping = tf.dtypes.cast(utils.task_conditionning(tf.dtypes.cast(task_mapping + 1, tf.float32)) - 1, tf.int32)
            else: 
                uniform_sampling = tf.random.uniform(shape=(batch_size, cur_leaf * branches, 1),
                                                     minval=0,
                                                     maxval=1)
                y_k_new = tf.dtypes.cast(pred_main - uniform_sampling > 0,
                                         dtype=tf.float32)
                y_k_new = 2 * tf.tile(y_k_new, (1, 1, self.T)) - 1

            # (B, cur_leaf * branches, T)
            task_mask = tf.gather(tf.eye(self.T), permutation, axis=0)


            # (B, cur_leaf * branches, T)

            y_k = (y_k + y_k_new * task_mask) / (tf.math.sqrt(tf.dtypes.cast(k, tf.float32)) + 1) 
            # y_k = y_k_new * task_mask


            cur_leaf = cur_leaf * branches

        permutations = tf.concat(permutations, axis=-1)
        permutation_matrices = tf.gather(tf.eye(self.T), permutations, axis=0)
        inv_permutation_matrices = tf.transpose(permutation_matrices, (0, 1, 3, 2))
        logits_main = tf.squeeze(tf.matmul(inv_permutation_matrices, tf.expand_dims(tf.concat(logits_main, axis=-1), axis=-1)), axis=-1)
        if training:
            mixtures = tf.concat(mixtures, axis=-1)

            logits_mixture = tf.concat(logits_mixture, axis=-1)
            logits_logsumexp = tf.concat(logits_logsumexp, axis=-1)
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['logits_mixture'] = logits_mixture
            outputs_dict['loss']['logits_logsumexp'] = logits_logsumexp
            outputs_dict['loss']['mixture'] = mixtures
            outputs_dict['loss']['logits_main'] = logits_main

        else:
            pred_main = tf.math.sigmoid(logits_main)
            outputs_dict['global_pred'] = tf.math.reduce_mean(pred_main, axis=1)
            # (B, T, T)
            mixtures = tf.concat(mixtures, axis=-1)
            outputs_dict['permutation_matrix'] = mixtures
            for k in range(self.T):
                outputs_dict['column_{}'.format(k)] = mixtures[:, k, :]

        return outputs_dict


 
class DropoutMaonet(tkm.Model):
    def __init__(self,
                 T,
                 dropouts,
                 recurrent_cell_args,
                 recurrent_mixture,
                 N_samples,
                 mixtX=False,
                 **kwargs):
        super(DropoutMaonet, self).__init__()
        self.dropouts = tf.constant(dropouts)
        self.N_samples = tf.constant(N_samples)

        self.T = T
        self.mixtX = mixtX
  
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type
        recurrent_cell_args['n_task'] = T

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.recurrent_mixture = recurrent_mixture 

        self.dense = PermutationDensev2(n_task=T,
                                        units=1,
                                        activation='linear')

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.label_encoder = tkl.Dense(units=64,
                                       activation='linear',
                                       kernel_initializer=previous_label_initializer)
        self.E = 64

        # Handling input compression
        self.input_compression = False
        self.H = recurrent_cell_args['units']
        self.input_compresser = tkl.Dense(units=self.H,
                                          activation='relu')



    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.H)
        super(DropoutMaonet, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        """
        inputs: float (B, I).
        y: float (B, T).
        """
        outputs_dict = dict()
        batch_size = tf.shape(inputs)[0]

        if self.input_compression:
            # (B, H)
            inputs = self.input_compresser(inputs, training=training)

        inputs = tf.expand_dims(inputs, axis=1)

        # (B, 1, H)
        state_mixture = inputs
        state_main = inputs

        if self.mixtX:
            state_mixture = inputs
        else:
            state_mixture = tf.zeros_like(inputs)

        mixtures = []
        logits_mixture = []
        logits_logsumexp = []
        logits_main = []
        permutations = []

        task_mapping = tf.range(self.T)
        T_factorial = tf.math.reduce_prod(task_mapping + 1)

        cur_leaf = 1
        y_k = tf.zeros((batch_size, cur_leaf, self.T))
        for k in range(self.T):
            dropout = self.dropouts[k]
            N_sample = self.N_samples[k]
            #####
            # Mixture
            # encoded_input : (B, T!/(T - k)!, E)
            if training:
                mixture_mask = (y_k == 0)

                mult_to_leafs = tf.math.reduce_prod(self.dropouts[k+1:])
                # branches = self.T - k
                # mult_to_leafs = tf.math.reduce_prod(tf.range(1, self.T - (k + 1) + 1))
                # total_leafs = T_factorial
                branches = dropout
                total_leafs = tf.math.reduce_prod(self.dropouts)

            else:
                mixture_mask = tf.dtypes.cast(y_k == 0, dtype=tf.float32)
                branches = N_sample
                mult_to_leafs = tf.math.reduce_prod(self.N_samples[k+1:])
                total_leafs = tf.math.reduce_prod(self.N_samples)
                # mult_to_leafs = tf.math.pow(self.N_sample, self.T - (k + 1))
                # total_leafs = tf.math.pow(self.N_sample, self.T)


            encoded_input = self.label_encoder(y_k)
            # output_mixture: (B, T!/(T - k)!, H), states_mixture: (B, T!/(T - k)!, H)
            (logit_mixture, state_mixture) = self.recurrent_mixture(inputs=encoded_input,
                                                                    states=[state_mixture],
                                                                    training=training)
            if training:
                logit_mixture = tf.boolean_mask(logit_mixture, mixture_mask)
                logit_mixture = tf.reshape(logit_mixture,
                                           (batch_size, cur_leaf, self.T - k))
                # (B * cur_leaf, dropout, 1)

                kept_permutation = tf.expand_dims(sample_without_replacement(logit_mixture, branches),
                                                  axis=-1)
                # kept_permutation = tf.expand_dims(tf.random.categorical(tf.ones((batch_size * cur_leaf, self.T - k)), dropout, dtype=tf.int32), axis=-1)
                kept_permutation = tf.reshape(kept_permutation,
                                              (batch_size, cur_leaf, dropout, 1))

                """
                kept_permutation = tf.expand_dims(sample_without_replacement(tf.ones((batch_size, cur_leaf, self.T - k)),
                                                                             self.dropout),
                                                  axis=-1)
                """

                # (B, cur_leaf)
                perm_index, batch_index = tf.meshgrid(tf.range(cur_leaf), tf.range(batch_size))
                # (B, cur_leaf, branches, 1)
                batch_index = tf.expand_dims(tf.tile(tf.expand_dims(batch_index, axis=-1), (1, 1, dropout)),
                                                     axis=-1)
                # (B, cur_leaf, branches, 1)
                perm_index = tf.expand_dims(tf.tile(tf.expand_dims(perm_index, axis=-1), (1, 1, dropout)),
                                            axis=-1)

                # (B, cur_leaf, branches)
                logit_mixture = tf.gather_nd(params=logit_mixture,
                                             indices=tf.concat([batch_index, perm_index, kept_permutation],
                                                               axis=-1))
                

                mixture = tf.nn.softmax(logit_mixture, axis=-1)
                mixture = tf.reshape(mixture,
                                     (batch_size, cur_leaf * branches))

                # (B, cur_leaf, branches)
                logit_logsumexp = tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(logit_mixture, axis=-1), -1),
                                          (1, 1, branches))



                # (B, cur_leaf * branches)
                logit_mixture = tf.reshape(logit_mixture,
                                           (batch_size, cur_leaf * branches))
                # (B, 1, cur_leaf * branches)
                logit_mixture = tf.expand_dims(logit_mixture, axis=1)
                # (B, mult_to_leafs, cur_leaf * branches)
                logit_mixture = tf.tile(logit_mixture, (1, mult_to_leafs, 1))
                # (B, total_leafs, 1)
                logit_mixture = tf.reshape(tf.transpose(logit_mixture, (0, 2, 1)),
                                           (batch_size, total_leafs, 1))
                logits_mixture.append(logit_mixture)


                # (B, cur_leaf * branches)
                logit_logsumexp = tf.reshape(logit_logsumexp,
                                             (batch_size, cur_leaf * branches))
                # (B, 1, cur_leaf * branches)
                logit_logsumexp = tf.expand_dims(logit_logsumexp, axis=1)
                # (B, mult_to_leafs, cur_leaf * branches)
                logit_logsumexp = tf.tile(logit_logsumexp, (1, mult_to_leafs, 1))
                # (B, total_leafs, 1)
                logit_logsumexp = tf.reshape(tf.transpose(logit_logsumexp, (0, 2, 1)),
                                             (batch_size, total_leafs, 1))
                logits_logsumexp.append(logit_logsumexp)


                # Duplicate logit mixture:
                # logit_mixture : (B, 1, T!/(T-(k + 1))!)
                mixture = tf.expand_dims(mixture, axis=1)
                mixture = tf.tile(mixture, (1, mult_to_leafs, 1))

                # logit_mixture : (B, T!, 1)
                mixture = tf.reshape(tf.transpose(mixture, (0, 2, 1)),
                                     (batch_size, total_leafs, 1))
                mixtures.append(mixture)
            else:
                # (B, cur_leaf, T)
                masked_exp_logit = mixture_mask * tf.math.exp(logit_mixture)
                logit_mixture = tf.math.log(masked_exp_logit)
                # (B, cur_leaf, T)
                mixture = masked_exp_logit / tf.tile(tf.expand_dims(tf.math.reduce_sum(masked_exp_logit, axis=-1), axis=-1), (1, 1, self.T))
                # (B, T)
                mixture = tf.math.reduce_mean(mixture, axis=1)
                mixtures.append(tf.expand_dims(mixture, axis=-1))
 
            # Fancy Tiling for next timestep.
            # (B, 1, cur_leaf, H)
            state_mixture = tf.expand_dims(state_mixture, axis=1)
            # (B, branches, cur_leaf, H)
            state_mixture = tf.tile(state_mixture, (1, branches, 1, 1))
            # (B, cur_leaf * branches, H)
            state_mixture = tf.reshape(tf.transpose(state_mixture, (0, 2, 1, 3)),
                                       (batch_size, cur_leaf * branches, self.H))


            #####
            # Main Network
            # Fancy tiling main input
            # (B, 1, cur_leaf, E)
            encoded_input = tf.expand_dims(encoded_input, axis=1)
            # (B, branches, cur_leaf, E)
            encoded_input = tf.tile(encoded_input, multiples=(1, branches, 1, 1))
            # (B, cur_leaf * branches, E)
            encoded_input = tf.reshape(tf.transpose(encoded_input, (0, 2, 1, 3)),
                                       (batch_size, cur_leaf * branches, self.E))


            # Fancy tiling main state
            # (B, 1, cur_leaf, H)
            state_main = tf.expand_dims(state_main, axis=1)
            # (B, branches, cur_leaf, H)
            state_main = tf.tile(state_main, (1, branches, 1, 1))
            # (B, cur_leaf * branches, H)
            state_main = tf.reshape(tf.transpose(state_main, (0, 2, 1, 3)),
                                    (batch_size, cur_leaf * branches, self.H))


            if training:
                # (B, cur_leaf, T)
                mixture_mask = (y_k == 0)
                permutation = tf.tile(tf.reshape(tf.range(self.T), (1, 1, self.T)),
                                      (batch_size, cur_leaf, 1))

                permutation = tf.reshape(tf.boolean_mask(permutation, mixture_mask),
                                         (batch_size, cur_leaf, self.T - k))

                permutation = tf.gather_nd(params=permutation,
                                           indices=tf.concat([batch_index, perm_index, kept_permutation],
                                                             axis=-1))
                permutation = tf.reshape(permutation, (batch_size, cur_leaf * branches))

                # tf.print('permutation2 : ', permutation)
                # Fancy tiling
                # (B, 1, cur_leaf * branches)
                permutation_tm = tf.expand_dims(permutation, axis=1)
                permutation_tm = tf.reshape(tf.transpose(tf.tile(permutation_tm,
                                                                     (1, mult_to_leafs, 1)),
                                                             (0, 2, 1)),
                                            (batch_size, total_leafs, 1))
            else:
                # (B * cur_leaf, T)
                logit_mixture = tf.reshape(logit_mixture, (batch_size * cur_leaf, self.T))
                # (B * cur_leaf, branches)
                permutation = tf.random.categorical(logit_mixture, branches)
                # (B, cur_leaf * branches)
                permutation = tf.reshape(permutation,
                                         (batch_size, cur_leaf * branches))
                permutation_tm = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(permutation, axis=1), (1, mult_to_leafs, 1)),
                                                         (0, 2, 1)),
                                            (batch_size, total_leafs, 1))

            permutations.append(permutation_tm)


            # (B, cur_leaf * branches, H)
            output_main, state_main = self.recurrent_cell(inputs=encoded_input,
                                                          states=state_main,
                                                          permutation=permutation,
                                                          training=training)

            # (B, cur_leaf * branches, 1)
            logit_main = self.dense(output_main,
                                    permutation=permutation,
                                    training=training)

            pred_main = tf.math.sigmoid(logit_main)

            # Duplicate logit_main :
            # logit_main : (B, 1, cur_leaf * branches, 1)
            logit_main = tf.expand_dims(logit_main, axis=1)
            # logit_main : (B, mult_to_leafs, cur_leaf * branches, 1)
            logit_main = tf.tile(logit_main, (1, mult_to_leafs, 1, 1))
            # logit_main : (B, total_leafs, 1)
            logit_main = tf.reshape(tf.transpose(logit_main, (0, 2, 1, 3)),
                                    (batch_size, total_leafs, 1))
                
            logits_main.append(logit_main)


            # Fancy tiling ground truth:
            # (B, 1, T! / (T - k)!, T)
            y_k = tf.expand_dims(y_k, axis=1)
            # (B, T - k, T!/(T-k)!, T) 
            y_k = tf.tile(y_k, (1, branches, 1, 1))
            y_k = tf.reshape(tf.transpose(y_k, (0, 2, 1, 3)),
                             (batch_size, cur_leaf * branches, self.T))


            if training:
                # (B, T!/(T - (k + 1))!)
                batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(batch_size), axis=-1),
                                                     (1, cur_leaf * branches)),
                                             axis=-1)
                # (B, T!/(T - (k + 1))!, 2)
                permutation_concat = tf.concat([batch_index, tf.expand_dims(permutation, axis=-1)], axis=-1)
                y_k_new = 2 * tf.gather_nd(params=y,
                                           indices=permutation_concat) - 1
                # (B, T!/(T - (k + 1))!, T)
                y_k_new = tf.tile(tf.expand_dims(y_k_new, axis=-1), (1, 1, self.T))

            else: 
                uniform_sampling = tf.random.uniform(shape=(batch_size, cur_leaf * branches, 1),
                                                     minval=0,
                                                     maxval=1)
                y_k_new = tf.dtypes.cast(pred_main - uniform_sampling > 0,
                                         dtype=tf.float32)
                y_k_new = 2 * tf.tile(y_k_new, (1, 1, self.T)) - 1

            # (B, cur_leaf * branches, T)
            task_mask = tf.gather(tf.eye(self.T), permutation, axis=0)


            # (B, cur_leaf * branches, T)

            y_k = (y_k + y_k_new * task_mask) 
            #/ (tf.math.sqrt(tf.dtypes.cast(k, tf.float32)) + 1)
            # y_k = y_k_new * task_mask


            cur_leaf = cur_leaf * branches

        permutations = tf.concat(permutations, axis=-1)
        permutation_matrices = tf.gather(tf.eye(self.T), permutations, axis=0)
        inv_permutation_matrices = tf.transpose(permutation_matrices, (0, 1, 3, 2))
        logits_main = tf.squeeze(tf.matmul(inv_permutation_matrices, tf.expand_dims(tf.concat(logits_main, axis=-1), axis=-1)), axis=-1)
        if training:
            mixtures = tf.concat(mixtures, axis=-1)

            logits_mixture = tf.concat(logits_mixture, axis=-1)
            logits_logsumexp = tf.concat(logits_logsumexp, axis=-1)
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['logits_mixture'] = logits_mixture
            outputs_dict['loss']['logits_logsumexp'] = logits_logsumexp
            outputs_dict['loss']['mixture'] = mixtures
            outputs_dict['loss']['logits_main'] = logits_main

        else:
            pred_main = tf.math.sigmoid(logits_main)
            outputs_dict['global_pred'] = tf.math.reduce_mean(pred_main, axis=1)
            # (B, T, T)
            mixtures = tf.concat(mixtures, axis=-1)
            outputs_dict['permutation_matrix'] = mixtures
            for k in range(self.T):
                outputs_dict['column_{}'.format(k)] = mixtures[:, k, :]

        return outputs_dict

##################################
# One Recurrent Cell task blocks #
##################################

class BlockPim2Seq(tkm.Model):
    def __init__(self,
                 blocks,
                 n_permutations,
                 permutation_heuristic,
                 vector,
                 recurrent_cell_args,
                 N_sample,
                 permutation_encoding=False,
                 **kwargs):

        super(BlockPim2Seq, self).__init__(**kwargs)

        self.n_task = sum([len(block) for block in blocks])
        self.n_blocks = len(blocks)

        """ All blocks should have the same size """
        self.len_blocks = len(blocks[0])

        self.N_sample = N_sample
        self.n_permutations = n_permutations

        #########################
        # Permutation Concerns
        ###############
        # Input Permutation
        # From [0, 1, ..., n_task - 1] to [B1, B2, ...., BN]
        concat_blocks = block_utils.sum_block(blocks, [])
        self.to_block_order = tf.Variable(np.identity(self.n_task)[concat_blocks, :],
                                          dtype=tf.float32,
                                          trainable=False)
        tf.print('to_block_order :\n ', self.to_block_order)
        ###############
        ###############
        # Block Permutation
        # From [B1, B2, ..., BN] to [Bs(1), Bs(2), ..., Bs(N)]

        permutation_sampler = permutation_heuristics.sample_with_heuristic(copy.deepcopy(permutation_heuristic))
        # (P, NB)
        block_permutations = permutation_sampler(self.n_permutations, self.n_blocks)
        self.block_mapping = tf.Variable(block_permutations,
                                         dtype=tf.int32,
                                         trainable=False)

        # (P, NB, NB)
        block_permutation_matrices = np.identity(self.n_blocks)[block_permutations]
        self.block_permutation_matrices = tf.Variable(block_permutation_matrices,
                                                      dtype=tf.float32,
                                                      trainable=False)
        permutation_matrices = []
        for i in range(self.n_permutations):
            # (1, T, T)
            permutation_matrices.append(block_utils.expand_to_blocks_v2(block_permutation_matrices[i, :, :],
                                                                        self.len_blocks)[np.newaxis, :, :])

        # (P, T, T)
        self.permutation_matrices = tf.Variable(np.concatenate(permutation_matrices, axis=0),
                                                dtype=tf.float32,
                                                trainable=False)

        #########################
        # Recurrent Cells
        self.units = recurrent_cell_args['units']
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation_{}'.format(recurrent_cell_type)
        recurrent_cell_args['n_task'] = self.n_blocks

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense1 = PermutationDense(n_task=self.n_task,
                                       units=2 * self.units,
                                       activation='relu')
        self.bn1 = tkl.BatchNormalization()
        self.dense2 = PermutationDense(n_task=self.n_task,
                                       units=self.len_blocks,
                                       activation='linear')


        # Handling input compression
        self.input_compression = False
        self.input_compresser = tkl.Dense(units=self.units,
                                          activation='relu')


        self.previous_label_encoder = tkl.Dense(units=64,
                                                activation='relu')

        #########################
        # Mixture Concerns
        self.vector = vector
 
    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.units)
        super(BlockPim2Seq, self).build(input_shape)

    def call(self, inputs, training=None, y=None, **kwargs):
        outputs_dict = dict()

        #########################
        # Mixture Concerns
        permutation_mixture_logits = self.vector(None)
        permutation_mixture = tf.nn.softmax(permutation_mixture_logits)

        perm_expend = tf.reshape(permutation_mixture, (self.n_permutations, 1, 1))
        perm_expend = tf.tile(perm_expend, multiples=[1, self.n_blocks, self.n_blocks])
        outputs_dict['mixture'] = permutation_mixture
        outputs_dict['permutation_matrix'] = tf.math.reduce_sum(perm_expend * self.block_permutation_matrices,
                                                                axis=0)

        batch_size = tf.shape(inputs)[0] 
        inputs = tf.expand_dims(inputs, axis=1)

        if training:
            parallel = self.n_permutations
            # (B, T, 1)
            y = tf.expand_dims(y, axis=-1)
            # Put y in the order that is in in the blocks : 
            # (B, T)
            y = tf.squeeze(tf.matmul(self.to_block_order, y), axis=-1)
            # (B, 1, T)
            y = tf.expand_dims(y, axis=1)
            # (B, P, T)
            y = tf.tile(y, multiples=[1, parallel, 1])
            # Put y in the order it is in the permutation
            y = tf.expand_dims(y, axis=-1)
            y = tf.squeeze(tf.matmul(self.permutation_matrices, y), axis=-1)
            y = tf.concat([tf.zeros((batch_size, self.n_permutations, 1)), y], axis=-1)
            permutation_samples = tf.range(0, self.n_permutations)

        else:
            parallel = self.N_sample
            permutation_samples = tf.squeeze(tf.random.categorical(tf.reshape(permutation_mixture_logits,
                                                                              (1, self.n_permutations)),
                                                                   num_samples=self.N_sample, dtype=tf.int32),
                                             axis=0)

        block_mapping = tf.gather(self.block_mapping, permutation_samples, axis=0)

        inputs = tf.tile(inputs, multiples=[1, parallel, 1])
        if self.input_compression:
            # (B, ?, E)
            inputs = self.input_compresser(inputs, training=training)

        padded_permutations = tf.concat([tf.zeros((self.n_permutations, 1, self.n_task)),
                                         self.permutation_matrices],
                                        axis=1)

        # (?, T + 1, T)
        padded_permutations = tf.gather(padded_permutations,
                                        permutation_samples,
                                        axis=0)

        padded_length = tf.concat([tf.ones((1, ), dtype=tf.int32), self.len_blocks * tf.ones((self.n_blocks, ), dtype=tf.int32)], axis=0)
        inv_permutation_matrices = tf.transpose(self.permutation_matrices, perm=(0, 2, 1))
        inv_to_block_order = tf.transpose(self.to_block_order, perm=(1, 0))

        #########################
        # (B, ?, E)
        states = inputs
        logits = []

        y_k = tf.zeros((batch_size, parallel, 1))
        block_cursor = 0

        for k in range(self.n_blocks):
            block_selection = tf.range(block_cursor, block_cursor + padded_length[k])
            if training:
                # (batchsize, P, blocksize)
                y_k = tf.gather(y, block_selection, axis=-1)

            # (batchsize, ?, blocksize)
            previous_label = y_k

            # (batchsize, ?, blocksize, 1)
            previous_label = tf.expand_dims(previous_label, axis=-1)

            # (batchsize, ?, blocksize, T)
            previous_label = tf.tile(2 * previous_label - 1,
                                     multiples=[1, 1, 1, self.n_task])

            # (1, ?, blocksize, T)
            projection = tf.reshape(tf.gather(padded_permutations, block_selection, axis=1),
                                    (1, parallel, padded_length[k], self.n_task))

            # (batch_size, ?, blocksize, T)
            projection = tf.tile(projection, multiples=[batch_size, 1, 1, 1])

            # (batch_size, ?, T)
            previous_label = tf.math.reduce_sum(previous_label * projection, axis=-2)

            # (batch_size, ?, E)
            previous_label = self.previous_label_encoder(previous_label, training=training)

            # cell_output, states : (batch_size, ?, H), (batch_size, ?, H)
            (cell_output, states) = self.recurrent_cell(permutation=block_mapping[:, k],
                                                        inputs=previous_label,
                                                        states=states,
                                                        training=training)

            # (batch_size, ?, 2 x H)
            cell_output = self.dense1(cell_output, permutation=block_mapping[:, k], training=training)
            cell_output = self.bn1(cell_output, training=training)

            # (batch_size, ?, blocksize)
            cell_output = self.dense2(cell_output, permutation=block_mapping[:, k], training=training)

            logits.append(cell_output)

            if not training:
                uniform_sampling = tf.random.uniform(shape=tf.shape(cell_output),
                                                     minval=0,
                                                     maxval=1)
                y_k = tf.dtypes.cast(tf.math.sigmoid(cell_output) - uniform_sampling > 0,
                                     dtype=tf.float32)

            block_cursor = block_cursor + padded_length[k]

        # (B, ?, T)
        logits = tf.concat(logits, axis=-1)

        # (B, ?, T, 1)
        logits = tf.expand_dims(logits, axis=-1)

        # (B, ?, T)
        # Put logits back in the original order
        logits = tf.squeeze(tf.matmul(inv_to_block_order, 
                                      tf.matmul(tf.gather(inv_permutation_matrices, permutation_samples, axis=0),
                                                logits)),
                            axis=-1)

        for i in range(parallel):
            key = 'permutation_wise_{}'.format(i)
            outputs_dict[key] = logits[:, i, :]

        if training:
            for k in range(self.n_task):
                key = "timestep_wise_{}".format(k)
                outputs_dict[key] = dict()
                outputs_dict[key]['prediction'] = logits[:, :, k]
                outputs_dict[key]['mixture'] = permutation_mixture

            outputs_dict['loss'] = dict()
            outputs_dict['loss']['mixture_logits'] = permutation_mixture_logits
            outputs_dict['loss']['mixture'] = permutation_mixture
            outputs_dict['loss']['output'] = logits

            mixture = permutation_mixture
            mixture = tf.reshape(mixture, (1, self.n_permutations, 1))
            mixture = tf.tile(mixture, multiples=[batch_size, 1, self.n_task])
        else:
            mixture = tf.ones((batch_size, self.N_sample, self.n_task))/self.N_sample
        
        # (B, T)
        outputs_dict['global_pred'] = tf.math.reduce_sum(tf.math.sigmoid(logits) * mixture, axis=1)
        return outputs_dict


SUPPORTED_PERMUTATION_REGRESSORS = {"sornet": SoftOrderingRecurrentRegressor,
                                    "ocpnet": OneCellPermutationNetwork,
                                    "pim2seq": PermutationIm2Seq,
                                    "parallele_pim2seq": ParallelizedPim2Seq,
                                    "spim2seq": StochasticPim2Seq,
                                    "dospim2seq": DropOutStochasticPim2Seq,
                                    "maonet": DropoutMaonet,
                                    "apim2seq": APim2Seq,
                                    "blockpernet": BlockPernet,
                                    "blockpim2seq": BlockPim2Seq,
                                    "hpernet": HierarchicalPernet,
                                    "pernet": PermutationRNN,
                                    "pernetv2": PermutationRNNv2,
                                    "pernetv3": Pernetv3,
                                    "subpernet": SubPermutationRNN,
                                    "T_spim2seq": T_StochasticPim2Seq}
