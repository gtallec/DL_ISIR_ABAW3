import tensorflow as tf
import numpy as np
import pandas as pd
import measures

from metrics_extended.abstract_metrics import SlidingMeanMetric, TensorSlidingMeanMetric, TensorTrackingMetric, DatasetMetric

from measures_extended.reweighted_measures import weighted_balanced_bce_permutation_loss, weighted_binary_permutation_loss, balanced_bce_permutation_loss

class SoftOrderTracking(TensorTrackingMetric):
    def __init__(self, T, softorder_in, log_folder, **kwargs):
        super(SoftOrderTracking, self).__init__(shape=(T, T),
                                                tensor_in=softorder_in,
                                                log_folder=log_folder)

class PermutationSelector(tf.keras.metrics.Metric):
    def __init__(self, n_permutations, name='permutation_losses', **kwargs):
        super(PermutationSelector, self).__init__(name=name)
        self.n_permutations = n_permutations

        self.centered_L_sigmas = self.add_weight(name='permutation_losses',
                                                 shape=(n_permutations, ),
                                                 initializer='zeros',
                                                 dtype=tf.float32)
        self.centered_L_sigmas_without_i = self.add_weight(name='permutation_losses',
                                                           shape=(n_permutations, n_permutations),
                                                           initializer='zeros',
                                                           dtype=tf.float32)
        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        permutation_mixture = y_pred.get('mixture')
        permutation_output = y_pred.get('output')

        M = tf.shape(y_true)[0]
        n_perm = tf.shape(permutation_mixture)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])
        # point wise bce of size (M, n_perm, T)
        pw_bce = measures.pointwise_bce(y_true_tiled, permutation_output)
        # permutation wise bce of size (M, n_perm)
        permw_bce = tf.math.reduce_mean(pw_bce,
                                        axis=-1)

        # (n_perm, )
        L_sigmas = tf.math.reduce_mean(permw_bce, axis=0)
        centered_L_sigmas = L_sigmas - tf.math.reduce_sum(L_sigmas * permutation_mixture) * tf.ones((n_perm, ))


        permutation_mixture_column = tf.tile(tf.expand_dims(permutation_mixture, -1), [1, n_perm])

        # (n_perm, )
        mean_L_without_i = tf.math.reduce_sum(permutation_mixture_column * (1 - tf.eye(n_perm)) * tf.tile(tf.expand_dims(L_sigmas, axis=-1),
                                                                                                          (1, n_perm)),
                                              axis=0)/(1 - permutation_mixture + tf.keras.backend.epsilon())

        centered_L_sigmas_without_i = (tf.tile(tf.expand_dims(L_sigmas, axis=-1), (1, n_perm))
                                       -
                                       tf.tile(tf.expand_dims(mean_L_without_i, axis=0), (n_perm, 1)))

        M = tf.dtypes.cast(M, dtype=tf.float32)

        # (n_perm, n_perm)
        self.centered_L_sigmas_without_i.assign(self.centered_L_sigmas_without_i * (self.N / (self.N + M))
                                                +
                                                centered_L_sigmas_without_i * (M / (self.N + M)))


        # (n_perm, )
        self.centered_L_sigmas.assign(self.centered_L_sigmas * (self.N / (self.N + M))
                                      +
                                      centered_L_sigmas * (M / (self.N + M)))
        self.N.assign_add(M)
    

    def reset_states(self):
        self.centered_L_sigmas_without_i.assign(tf.zeros((self.n_permutations, self.n_permutations)))
        self.centered_L_sigmas.assign(tf.zeros((self.n_permutations, )))
        self.N.assign(0)


    def result(self):
        L_sigma_below = (self.centered_L_sigmas <= 0)
        n_permutations = tf.shape(L_sigma_below)[0]
        L_sigma_below = tf.tile(tf.expand_dims(L_sigma_below, axis=-1), (1, n_permutations))
        L_sigma_without_i_above = (self.centered_L_sigmas_without_i >= 0)
        c_mask = tf.math.reduce_any(tf.math.logical_and(L_sigma_below, L_sigma_without_i_above),
                                    axis=0)
        mask = tf.dtypes.cast(tf.logical_not(c_mask), dtype=tf.float32)
        return mask

class MeanFrobNormToIdentity(SlidingMeanMetric):
    def __init__(self, T, **kwargs):
        (super(MeanFrobNormToIdentity, self)
         .__init__(name='frobnorm',
                   eval_function=measures.mean_frobnorm_to_mat(mat=tf.eye(T))))

class PermutationLosses(TensorSlidingMeanMetric):
    def __init__(self, pred_in, P, name='permutation_losses', **kwargs):
        (super(PermutationLosses, self)
         .__init__(name='permutation_losses',
                   shape=(P, ),
                   eval_function=measures.permutation_losses(pred_in)))
        self.P = P
        self.column_name = name

    def result_to_df(self):
        result_columns = [self.column_name + '_' + str(i) for i in range(self.P)]
        result = self.result().numpy().reshape(1, self.P)
        return pd.DataFrame(data=result, columns=result_columns)


class PermutationLoss(tf.keras.metrics.Metric):
    def __init__(self, name='permutation_loss', **kwargs):
        super(PermutationLoss, self).__init__(**kwargs)
        self.moving_bce = self.add_weight(name='moving_bce',
                                          initializer='zeros')
        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        permutation_mixture = y_pred.get('mixture')
        permutation_output = y_pred.get('output')

        M = tf.shape(y_true)[0]
        n_perm = tf.shape(permutation_mixture)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])
        # point wise bce of size (M, n_perm, T)
        pw_bce = measures.pointwise_bce(y_true_tiled, permutation_output)
        # permutation wise bce of size (M, n_perm)
        permw_bce = tf.math.reduce_sum(pw_bce,
                                       axis=-1)
        tiled_mixture = tf.tile(tf.expand_dims(permutation_mixture, 0),
                                multiples=[M, 1])
        # element wise bce of size (M,)
        elw_bce = tf.math.reduce_sum(permw_bce * tiled_mixture, axis=1)
        batch_bce = tf.math.reduce_mean(elw_bce)
        M = tf.dtypes.cast(M, dtype=tf.float32)

        combination_coeff = self.N / (self.N + M)
        
        self.moving_bce = (self.moving_bce * combination_coeff
                           +
                           batch_bce * (1 - combination_coeff))
        self.N = self.N + M

    def result(self):
        return 0

class DropoutPermutationLoss(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(DropoutPermutationLoss, self)
         .__init__(name='dropout_permutation_loss',
                   eval_function=measures.dropout_permutation_loss))

class PermutationMeanMetric(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(PermutationMeanMetric, self)
         .__init__(name='perm_mean',
                   eval_function=measures.permutation_mean))

class PermutationVarMetric(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(PermutationVarMetric, self)
         .__init__(name='perm_var',
                   eval_function=measures.permutation_variance))


class TaskPermutationLosses(TensorSlidingMeanMetric):
    def __init__(self, T, pred_in, dataset_columns, name='task_permutation_losses', **kwargs):
        TensorSlidingMeanMetric.__init__(self, name=name, eval_function=measures.task_permutation_losses(pred_in), shape=(T, ))
        self.columns = [name + dataset_column for dataset_column in dataset_columns]
        # DatasetMetric.__init__(self, dataset_columns=dataset_columns, name=name, **kwargs)

    def result_to_df(self):
        # (1, T)
        result = tf.expand_dims(self.result(), axis=0).numpy()
        return pd.DataFrame(data=result, columns=self.columns)


class FrobNormToIdentity(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        (super(FrobNormToIdentity, self)
         .__init__(**kwargs))
        
        self.fro = self.add_weight(name='frobNorm',
                                   initializer='zeros',
                                   dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        T = tf.shape(y_pred)[0]
        self.fro.assign(measures.frob_norm(y_pred - tf.eye(T)))

    def result(self):
        return self.fro

class FrobNormToPermutation(tf.keras.metrics.Metric):
    def __init__(self, permutation, **kwargs):
        (super(FrobNormToPermutation, self)
         .__init__())

        self.fro = self.add_weight(name='frobNorm_to_permutation',
                                   initializer='zeros',
                                   dtype=tf.float32)
        self.permutation = permutation

    def update_state(self, y_true, y_pred, sample_weight=None):
        T = tf.shape(y_pred)[0]
        permutation_matrix = tf.gather(tf.eye(T), self.permutation)
        self.fro.assign(measures.frob_norm(y_pred - permutation_matrix))

    def result(self):
        return self.fro

class BlockFrobNormToIdentity(tf.keras.metrics.Metric):
    def __init__(self, block, **kwargs):
        (super(BlockFrobNormToIdentity, self)
         .__init__(**kwargs))
        
        self.fro = self.add_weight(name='frobNorm',
                                   initializer='zeros',
                                   dtype=tf.float32)
        self.block = block

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_block = tf.gather(tf.gather(y_pred, self.block, axis=0), self.block, axis=1) 
        self.fro.assign(measures.frob_norm(y_block - tf.eye(len(self.block))))

    def result(self):
        return self.fro
 
class DropoutPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(DropoutPermutationLoss, self)
         .__init__(name='dropout_permutation_loss',
                   eval_function=measures.dropout_permutation_loss(pred_in)))

class DropoutJensenPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(DropoutJensenPermutationLoss, self)
         .__init__(name='dropout_jensen_permutation_loss',
                   eval_function=measures.dropout_jensen_permutation_loss(pred_in)))

class TreePermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(TreePermutationLoss, self)
         .__init__(name='tree_permutation_loss',
                   eval_function=measures.tree_permutation_loss(pred_in)))

class SSPCLoss(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(SSPCLoss, self)
         .__init__(name='sspc_loss_metrc',
                   eval_function=measures.semi_supervised_permutation_categorical_loss(pred_in)))

class CategoricalPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(CategoricalPermutationLoss, self)
         .__init__(name='cp_loss',
                   eval_function=measures.permutation_categorical_loss(pred_in)))

class WeightedCategoricalPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, task_weights, name, **kwargs):
        (super(WeightedCategoricalPermutationLoss, self)
         .__init__(name=name,
                   eval_function=measures.weighted_permutation_categorical_loss(pred_in, task_weights)))

class Bp4dAUCategoricalPermutationLoss(WeightedCategoricalPermutationLoss):
    def __init__(self, pred_in, occurences, name='bp4d-au_cp_loss', **kwargs):
        n_coords = tf.shape(occurences)[0]
        AU_occurences = tf.reshape(occurences, (n_coords // 2, 2))[:, 0]
        AU_w = (1 / AU_occurences) / tf.math.reduce_sum(1 / AU_occurences)
        super(Bp4dAUCategoricalPermutationLoss, self).__init__(pred_in=pred_in,
                                                               task_weights=AU_w,
                                                               name=name,
                                                               **kwargs)
class WeightedCategoricalPermutationDice(SlidingMeanMetric):
    def __init__(self, pred_in, task_weights, name="wcp_dice", **kwargs):
        (super(WeightedCategoricalPermutationDice, self)
         .__init__(name=name,
                   eval_function=measures.weighted_categorical_permutation_dice(pred_in, task_weights)))

class Bp4dAUPermutationDice(WeightedCategoricalPermutationDice):
    def __init__(self, pred_in, occurences, name='bp4d-au_wp_dice', **kwargs):
        n_coords = tf.shape(occurences)[0]
        AU_occurences = tf.reshape(occurences, (n_coords // 2, 2))[:, 0]
        AU_w = (1 / AU_occurences) / tf.math.reduce_sum(1 / AU_occurences)
        super(Bp4dAUPermutationDice, self).__init__(pred_in=pred_in,
                                                    task_weights=AU_w,
                                                    name=name,
                                                    **kwargs)

class Bp4dAUandSexPermutationDice(WeightedCategoricalPermutationDice):
    def __init__(self, pred_in, occurences, name='bp4d-au_sex_dice', **kwargs):
        n_coords = tf.shape(occurences)[0]
        positive_occurences = tf.reshape(occurences, (n_coords // 2, 2))[:, 0]
        positive_w = (1 / positive_occurences) / tf.math.reduce_sum(1 / positive_occurences)
        super(Bp4dAUandSexPermutationDice, self).__init__(pred_in=pred_in,
                                                          task_weights=positive_w,
                                                          name=name,
                                                          **kwargs)

class Bp4dAUandSexCategoricalPermutationLoss(WeightedCategoricalPermutationLoss):
    def __init__(self, pred_in, occurences, name='bp4d-au_sex_cp_loss', **kwargs):
        n_coords = tf.shape(occurences)[0]
        positive_occurences = tf.reshape(occurences, (n_coords // 2, 2))[:, 0]
        positive_w = (1 / positive_occurences) / tf.math.reduce_sum(1 / positive_occurences)
        super(Bp4dAUandSexCategoricalPermutationLoss, self).__init__(pred_in=pred_in,
                                                                     task_weights=positive_w,
                                                                     name=name,
                                                                     **kwargs)

class WeightedBinaryPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, task_weights, name="wbp_loss", **kwargs):
        (super(WeightedBinaryPermutationLoss, self)
         .__init__(name=name,
                   eval_function=measures.weighted_binary_permutation_loss(pred_in, task_weights)))

class WeightedPermutationDice(SlidingMeanMetric):
    def __init__(self, pred_in, task_weights, name="wp_dice", **kwargs):
        (super(WeightedPermutationDice, self)
         .__init__(name=name,
                   eval_function=measures.weighted_binary_permutation_dice(pred_in, tf.constant(task_weights,
                                                                                                dtype=tf.float32))))

class FrequencyWeightedBinaryPermutationLoss(WeightedBinaryPermutationLoss):
    def __init__(self, pred_in, occurences, name='fwbp_loss', **kwargs):
        task_weights = (1 / occurences) / tf.math.reduce_sum(1 / occurences)
        super(FrequencyWeightedBinaryPermutationLoss, self).__init__(pred_in=pred_in,
                                                                     task_weights=task_weights,
                                                                     name=name,
                                                                     **kwargs)

class SoftFrequencyWeightedBinaryPermutationLoss(WeightedBinaryPermutationLoss):
    def __init__(self, pred_in, frequencies, name='sfwbp_loss', **kwargs):
        task_weights = (1 - frequencies) / tf.math.reduce_sum(1 - frequencies)
        super(SoftFrequencyWeightedBinaryPermutationLoss, self).__init__(pred_in=pred_in,
                                                                         task_weights=task_weights,
                                                                         name=name,
                                                                         **kwargs)

class FrequencyWeightedPermutationDice(WeightedPermutationDice):
    def __init__(self, pred_in, frequencies, name='fwp_dice', **kwargs):
        task_weights = (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FrequencyWeightedPermutationDice, self).__init__(pred_in=pred_in,
                                                               task_weights=task_weights,
                                                               name=name,
                                                               **kwargs)

class SoftFrequencyWeightedPermutationDice(WeightedPermutationDice):
    def __init__(self, pred_in, frequencies, name='sfwp_dice', **kwargs):
        task_weights = (1 - frequencies) / tf.math.reduce_sum(1 - frequencies)
        super(SoftFrequencyWeightedPermutationDice, self).__init__(pred_in=pred_in,
                                                                   task_weights=task_weights,
                                                                   name=name,
                                                                   **kwargs)
class BalancedBCEPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, weights_positive, weights_negative, name='bbcep_loss', **kwargs):
        super(BalancedBCEPermutationLoss, self).__init__(eval_function=balanced_bce_permutation_loss(pred_in,
                                                                                                     weights_positive=weights_positive,
                                                                                                     weights_negative=weights_negative),
                                                         name=name,
                                                         **kwargs)

class FrequencyBalancedBCEPermutationLoss(BalancedBCEPermutationLoss):
    def __init__(self, pred_in, frequencies, name='fbbcep_loss', **kwargs):
        weights_positive = 1 - frequencies
        weights_negative = frequencies 
        super(FrequencyBalancedBCEPermutationLoss, self).__init__(pred_in=pred_in,
                                                                  name=name,
                                                                  weights_positive=weights_positive,
                                                                  weights_negative=weights_negative,
                                                                  **kwargs)

class InverseFrequencyBalancedBCEPermutationLoss(BalancedBCEPermutationLoss):
    def __init__(self, pred_in, frequencies, name='ifbbcep_loss', **kwargs):
        weights_positive = 1 / frequencies
        weights_negative = 1 / (1 - frequencies)
        super(InverseFrequencyBalancedBCEPermutationLoss, self).__init__(pred_in,
                                                                         weights_positive=weights_positive,
                                                                         weights_negative=weights_negative,
                                                                         name=name,
                                                                         **kwargs)

class WeightedBalancedBCEPermutationLoss(SlidingMeanMetric):
    def __init__(self, pred_in, task_weights, weights_positive, weights_negative, name='wbbcep_loss', **kwargs):
        super(WeightedBalancedBCEPermutationLoss, self).__init__(eval_function=weighted_balanced_bce_permutation_loss(pred_in=pred_in,
                                                                                                                      weights_positive=weights_positive,
                                                                                                                      weights_negative=weights_negative,
                                                                                                                      task_weights=task_weights),
                                                                 name=name)

class FrequencyWeightedBalancedBCEPermutationLoss(WeightedBalancedBCEPermutationLoss):
    def __init__(self, pred_in, frequencies, name='fwbbcep_loss', **kwargs):
        weights_positive = 1 / frequencies
        weights_negative = 1 / (1 - frequencies)
        task_weights = weights_positive / tf.math.reduce_sum(weights_positive)
        super(FrequencyWeightedBalancedBCEPermutationLoss, self).__init__(pred_in=pred_in,
                                                                          weights_positive=weights_positive,
                                                                          weights_negative=weights_negative,
                                                                          task_weights=task_weights,
                                                                          name=name)


SUPPORTED_PERMUTATION_METRICS = {"perm_selector": PermutationSelector,
                                 "do_perm_loss": DropoutPermutationLoss,
                                 "do_jensen_perm": DropoutJensenPermutationLoss,
                                 "tree_perm_loss": TreePermutationLoss,
                                 "softorder": SoftOrderTracking,
                                 "sspc_loss": SSPCLoss,
                                 "cp_loss": CategoricalPermutationLoss,
                                 "wcp_loss": WeightedCategoricalPermutationLoss,
                                 "perm_loss": PermutationLoss,
                                 "perm_losses": PermutationLosses,
                                 "perm_mean": PermutationMeanMetric,
                                 "perm_var": PermutationVarMetric,
                                 "mfrob2Id": MeanFrobNormToIdentity,
                                 # "frob2Id": FrobNormToIdentity,
                                 "frob2perm": FrobNormToPermutation,
                                 "blockfrob2Id": BlockFrobNormToIdentity,
                                 "fwbp_loss": FrequencyWeightedBinaryPermutationLoss,
                                 "sfwbp_loss": SoftFrequencyWeightedBinaryPermutationLoss,
                                 "fbbcep_loss": FrequencyBalancedBCEPermutationLoss,
                                 "fwbbcep_loss": FrequencyWeightedBalancedBCEPermutationLoss,
                                 "ifbbcep_loss": InverseFrequencyBalancedBCEPermutationLoss,
                                 "wp_dice": WeightedPermutationDice,
                                 "fwp_dice": FrequencyWeightedPermutationDice,
                                 "sfwp_dice": SoftFrequencyWeightedPermutationDice,
                                 "bp4d-au_cp_loss": Bp4dAUCategoricalPermutationLoss,
                                 "bp4d-au_dice": Bp4dAUPermutationDice,
                                 "bp4d-au_sex_cp_loss": Bp4dAUandSexCategoricalPermutationLoss,
                                 "bp4d-au_sex_dice": Bp4dAUandSexPermutationDice,
                                 "task_loss": TaskPermutationLosses}
