from models.layers.recurrent_cells import recurrent_cell
from models.layers.permutation_layers import PermutationDensev2, PermutationDenseCategorical
from utils import sample_without_replacement

import tensorflow.keras.layers as tkl
import tensorflow_probability as tfp
import tensorflow.keras.models as tkm
import tensorflow as tf

import copy



class Damonetv1(tkm.Model):
    def __init__(self,
                 n_classes,
                 dropouts,
                 task_controller,
                 pred_units,
                 label_units,
                 recurrent_cell_args,
                 mixtX,
                 N_samples,
                 **kwargs):
        super(Damonetv1, self).__init__()
        self.dropouts = tf.constant(dropouts)
        self.N_samples = tf.constant(N_samples)

        self.T = len(n_classes)
        self.T_tot = sum(n_classes)
        self.mixtX = mixtX
  
        recurrent_cell_type = recurrent_cell_args.pop('type')
        recurrent_cell_args['type'] = 'permutation' + '_' + recurrent_cell_type

        # A recurrent cell per task + 1 recurrent for the initialization.
        recurrent_cell_args['n_task'] = self.T + 1

        self.recurrent_cell = recurrent_cell(recurrent_cell_args)

        self.prediction_dense1 = PermutationDensev2(n_task=self.T,
                                                    units=pred_units,
                                                    activation='relu')
        self.prediction_dense2 = PermutationDenseCategorical(n_classes=n_classes,
                                                             activation='linear')

        self.task_controller = task_controller

        self.H_pred = recurrent_cell_args['units']
        self.H_ctrl = self.task_controller.get_units()
       

        previous_label_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.label_encoder_pred = tkl.Dense(units=label_units,
                                            activation='linear',
                                            kernel_initializer=previous_label_initializer)
        self.label_encoder_ctrl = tkl.Dense(units=label_units,
                                            activation='linear',
                                            kernel_initializer=previous_label_initializer)
        
        self.input_compresser_ctrl = tkl.Dense(units=self.H_pred,
                                               activation='relu')
        self.input_compresser_pred = tkl.Dense(units=self.H_ctrl,
                                               activation='relu')

        task_mask = []
        for i in range(len(n_classes)):
            task_mask_i = tf.concat([tf.zeros((i, n_classes[i])), tf.ones((1, n_classes[i])), tf.zeros((self.T - (i + 1), n_classes[i]))], axis=0)
            task_mask.append(task_mask_i)
        self.task_mask = tf.concat(task_mask, axis=1)


    """
    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.H)
        super(Damonet, self).build(input_shape)
    """

    def call(self, inputs, training=None, y=None, **kwargs):
        # inputs: float (B, I).
        # y: float (B, T).
        outputs_dict = dict()
        B = tf.shape(inputs)[0]

        # (B, H_ctrl)
        if self.mixtX:
            states_ctrl_t = self.input_compresser_ctrl(inputs, training=training)
        else:
            states_ctrl_t = tf.zeros((B, self.H_ctrl))
        
        # (B, H_pred)
        states_pred_t = self.input_compresser_pred(inputs, training=training)
            
        # Initialization (t = 0)
        # (B, 1, H_pred)
        states_pred_t = tf.expand_dims(states_pred_t, axis=1)
        states_ctrl_t = tf.expand_dims(states_ctrl_t, axis=1)
        # (B, 1, T)
        task_summary_t = tf.zeros((B, 1, self.T))
        y_t = tf.zeros((B, 1, self.T_tot), dtype=tf.float32)
        # (B, 1)
        tasks_t = tf.zeros((B, 1), dtype=tf.int32)
        leaf_t = 1

        if training:
            total_leafs = tf.math.reduce_prod(self.dropouts)
        else:
            total_leafs = tf.math.reduce_prod(self.N_samples)

        # Recurrent Loop:
        logits_pred = tf.zeros((B, total_leafs, self.T_tot))
        logits_ctrl = []
        columns = []
        logits_ctrl_logsumexp = []
        
        for t in range(self.T + 1):
            input_ctrl_t = self.label_encoder_ctrl(y_t, training=training)

            #########################
            # PREDICTION FOR TASK AT TIMESTEP T :
            if t > 0:
                input_pred_t = self.label_encoder_pred(y_t, training=training)

                # (B, leaf_t, T_tot), (B, leaf_t, T_tot)
                output_pred_t, states_pred_t = self.recurrent_cell(inputs=input_pred_t,
                                                                   states=states_pred_t,
                                                                   permutation=tasks_t)
                output_pred_t = self.prediction_dense1(output_pred_t,
                                                       permutation=tasks_t,
                                                       training=training)
                pred_t, task_mask_t = self.prediction_dense2(output_pred_t,
                                                             permutation=tasks_t,
                                                             training=training)

                
                if training:
                    # TODO : Harvest predictions in a by permutation design.
                    # (B, leaf_t, T_tot) : ground_truth for the treated tasks.
                    y_t = tf.tile(tf.expand_dims(y, axis=1), (1, leaf_t, 1)) * task_mask_t
                else:
                    # Sample one-hot vectors from pred_t
                    # (B, leaf_t, T_tot)
                    exp_pred_t = tf.math.exp(pred_t) * task_mask_t
                    p_t = exp_pred_t / tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_pred_t, axis=-1), axis=-1),
                                               (1, 1, self.T_tot))
                    # (B, leaf_t, T_tot)
                    y_t = tfp.distributions.OneHotCategorical(probs=p_t, dtype=tf.float32).sample()

                # (B, leaf_t, T)
                task_summary = tf.gather(tf.eye(self.T), tasks_t, axis=0)
                # (B, leaf_t, T)
                task_summary_t = task_summary_t + task_summary

                # (B, 1, leaf_t, T_tot)
                pred_t = tf.expand_dims(pred_t, axis=1)
                # (B, branches, leaf_t, T_tot)
                pred_t = tf.tile(pred_t, (1, mult_to_leafs, 1, 1))
                # (B, branches * leaf_t, T_tot)
                pred_t = tf.reshape(tf.transpose(pred_t, (0, 2, 1, 3)),
                                    (B, leaf_t * mult_to_leafs, self.T_tot))

                logits_pred = logits_pred + pred_t
                # tf.print(pred)


            # Mask for mixture computation: True for the not yet computed tasks.
            mask_t = (task_summary_t == 0)
            n_processed_t = tf.dtypes.cast(tf.math.reduce_mean(tf.math.reduce_sum(task_summary_t, axis=2)),
                                           dtype=tf.int32)

            #########################
            # NEXT TIMESTEP TASK SELECTION :
            if t < self.T:
                dropout = self.dropouts[t]
                N_sample = self.N_samples[t]

                if training:
                    mult_to_leafs = tf.math.reduce_prod(self.dropouts[t+1:])
                    branches = dropout
                else:
                    mult_to_leafs = tf.math.reduce_prod(self.N_samples[t+1:])
                    branches = N_sample
           
                input_ctrl_t = self.label_encoder_ctrl(y_t, training=training)
                # (B, leaf_t, T)
                logit_ctrl_t, states_ctrl_t = self.task_controller(inputs=input_ctrl_t,
                                                                   states=states_ctrl_t)
                # Masking the tasks that have already been processed
                logits_ctrl_t_masked = tf.reshape(tf.boolean_mask(logit_ctrl_t, mask_t),
                                                  (B, leaf_t, self.T - n_processed_t))
                permutation = tf.tile(tf.reshape(tf.range(self.T), (1, 1, self.T)),
                                      (B, leaf_t, 1))
                permutation_masked = tf.reshape(tf.boolean_mask(permutation, mask_t),
                                                (B, leaf_t, self.T - n_processed_t))

                if training:
                    # DROPOUT : Uniform sampling of branches tasks without replacement. 
                    # (B, leaf_t, branches, 1) : kept task in term of masked indices
                    kept_permutations = tf.expand_dims(sample_without_replacement(tf.ones_like(logits_ctrl_t_masked),
                                                                                  branches),
                                                       axis=-1)
                else:
                    # INFERENCE : Sampling of branches tasks based on the logit mixture coefficients
                    # (B, leaf_t, branches, 1) : kept task in term of masked indices
                    kept_permutations = tf.reshape(tf.random.categorical(tf.reshape(logits_ctrl_t_masked,
                                                                                    (B * leaf_t, self.T - n_processed_t)),
                                                                         branches,
                                                                         dtype=tf.int32),
                                                   (B, leaf_t, branches, 1))

                # Convert kept task from masked indices to global task index.
                # (B, leaf_t), (B, leaf_t)
                perm_index, batch_index = tf.meshgrid(tf.range(leaf_t), tf.range(B))
                # (B, leaf_t, branches, 1), (B, leaf_t, branches, 1)
                batch_index = tf.expand_dims(tf.tile(tf.expand_dims(batch_index, -1), (1, 1, branches)),
                                             axis=-1)
                perm_index = tf.expand_dims(tf.tile(tf.expand_dims(perm_index, -1), (1, 1, branches)),
                                            axis=-1)
                # (B, leaf_t, branches, 3)
                kept_permutations = tf.concat([batch_index, perm_index, kept_permutations], axis=-1)

                # (B, leaf_t, branches)
                logits_ctrl_t = tf.gather_nd(params=logits_ctrl_t_masked,
                                             indices=kept_permutations)

                logits_ctrl_logsumexp_t = tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(logits_ctrl_t, axis=-1), -1),
                                                  (1, 1, branches))


                # Build column of logit matrices:
                # (B, leaf_t * branches)
                logits_ctrl_t = tf.reshape(logits_ctrl_t, (B, leaf_t * branches))
                logits_ctrl_logsumexp_t = tf.reshape(logits_ctrl_logsumexp_t, (B, leaf_t * branches))

                # (B, 1, leaf_t * branches)
                logits_ctrl_t = tf.expand_dims(logits_ctrl_t, axis=1)
                logits_ctrl_logsumexp_t = tf.expand_dims(logits_ctrl_logsumexp_t, axis=1)

                # (B, mult_to_leafs, leaf_t * branches)
                logits_ctrl_t = tf.tile(logits_ctrl_t, (1, mult_to_leafs, 1))
                logits_ctrl_logsumexp_t = tf.tile(logits_ctrl_logsumexp_t, (1, mult_to_leafs, 1))

                # (B, total_leafs, 1) 
                logits_ctrl_t = tf.reshape(tf.transpose(logits_ctrl_t, (0, 2, 1)),
                                           (B, total_leafs, 1))
                logits_ctrl_logsumexp_t = tf.reshape(tf.transpose(logits_ctrl_logsumexp_t, (0, 2, 1)),
                                                     (B, total_leafs, 1))

                logits_ctrl.append(logits_ctrl_t)
                logits_ctrl_logsumexp.append(logits_ctrl_logsumexp_t)

             
                # (B, leaf_t, branches)
                tasks_t = tf.gather_nd(params=permutation_masked,
                                       indices=kept_permutations)
                tasks_t_tiled = tf.reshape(tasks_t, (B, 1, leaf_t * branches))
                tasks_t_tiled = tf.tile(tasks_t_tiled, (1, mult_to_leafs, 1))
                tasks_t_tiled = tf.reshape(tf.transpose(tasks_t_tiled, (0, 2, 1)),
                                           (B, total_leafs, 1))
                columns.append(tasks_t_tiled) 

                #########################
                # PREPARATION FOR NEXT TIMESTEP
                # Each leaf gives birth to branches branches
                # (B, 1, leaf_t, H_ctrl)
                states_ctrl_t = tf.expand_dims(states_ctrl_t, axis=1)
                # (B, branches, leaf_t, H)
                states_ctrl_t = tf.tile(states_ctrl_t, (1, branches, 1, 1))
                # (B, branches * leaf_t, H)
                states_ctrl_t = tf.reshape(tf.transpose(states_ctrl_t, (0, 2, 1, 3)),
                                           (B, leaf_t * branches, self.H_ctrl))

                # (B, 1, leaf_t, H)
                states_pred_t = tf.expand_dims(states_pred_t, axis=1)
                # (B, branches, leaf_t, H)
                states_pred_t = tf.tile(states_pred_t, (1, branches, 1, 1))
                # (B, branches * leaf_t, H)
                states_pred_t = tf.reshape(tf.transpose(states_pred_t, (0, 2, 1, 3)),
                                           (B, leaf_t * branches, self.H_pred))
                # (B, 1, leaf_t, T_tot)
                y_t = tf.expand_dims(y_t, axis=1)
                # (B, branches, leaf_t, T_tot)
                y_t = tf.tile(y_t, (1, branches, 1, 1))
                # (B, branches * leaf_t, T_tot)
                y_t = tf.reshape(tf.transpose(y_t, (0, 2, 1, 3)),
                                 (B, leaf_t * branches, self.T_tot))


                # (B, 1, leaf_t, T)
                task_summary_t = tf.expand_dims(task_summary_t, axis=1)
                # (B, branches, leaf_t, T)
                task_summary_t = tf.tile(task_summary_t, (1, branches, 1, 1))
                # (B, branches * leaf_t, T_tot)
                task_summary_t = tf.reshape(tf.transpose(task_summary_t, (0, 2, 1, 3)),
                                            (B, leaf_t * branches, self.T))
 
                tasks_t = tf.reshape(tasks_t, (B, leaf_t * branches))   
                leaf_t = leaf_t * branches

        logits_ctrl = tf.concat(logits_ctrl, axis=-1)
        logits_ctrl_logsumexp = tf.concat(logits_ctrl_logsumexp, axis=-1)
        # (B, total_leafs, T)
        permutation_matrix = tf.concat(columns, axis=-1)
        permutation_matrix = tf.math.reduce_mean(tf.one_hot(permutation_matrix, axis=-1, depth=self.T),
                                                 axis=1)
        # mixture = tf.math.exp(logits_ctrl - logits_ctrl_logsumexp)

        if training:
            outputs_dict['loss'] = dict()
            outputs_dict['loss']['log_mixture'] = logits_ctrl - logits_ctrl_logsumexp
            outputs_dict['loss']['logits_pred'] = logits_pred
            outputs_dict['loss']['task_mask'] = self.task_mask
        else:
            # (B, total_leafs, T, T_tot)
            logits_pred = tf.tile(tf.expand_dims(logits_pred, axis=-2), (1, 1, self.T, 1))
            # (B, total_leafs, T, T_tot)
            task_mask = tf.tile(tf.reshape(self.task_mask, (1, 1, self.T, self.T_tot)),
                                (B, total_leafs, 1, 1))
                
            exp_logits_pred = tf.math.exp(logits_pred) * task_mask
            # (B, total_leafs, T, T_tot)
            pred = exp_logits_pred / tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_logits_pred, axis=-1),
                                                            axis=-1),
                                             (1, 1, 1, self.T_tot))

            pred = tf.math.reduce_mean(tf.math.reduce_sum(pred, axis=-2),
                                       axis=1)
            # (B, T_tot)
            outputs_dict['global_pred'] = pred
            # (B, T, T_tot)
            pred_tiled = tf.tile(tf.expand_dims(pred, axis=1), (1, self.T, 1))
            # (B, T, T_tot)
            task_mask = tf.tile(tf.reshape(self.task_mask, (1, self.T, self.T_tot)),
                                (B, 1, 1))
            # (B, T_tot)
            pred_tiled = tf.math.reduce_sum(tf.one_hot(tf.math.argmax(pred_tiled * task_mask, axis=-1),
                                                       axis=-1,
                                                       depth=self.T_tot),
                                            axis=1)

            outputs_dict['onehot_pred'] = pred_tiled

            outputs_dict['soft_orders'] = permutation_matrix

        return outputs_dict 


SUPPORTED_DAMONETS = {"damonetv1": Damonetv1}
