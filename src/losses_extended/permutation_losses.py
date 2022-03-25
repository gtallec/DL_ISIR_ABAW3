import tensorflow as tf
import measures
import reducers

from losses_extended.abstract_losses import MeasureLoss
from measures_extended.reweighted_measures import weighted_balanced_bce_permutation_loss, weighted_binary_permutation_loss, balanced_bce_permutation_loss 

class PermutationLoss(tf.keras.losses.Loss):
    """ Computes weighted average of each permutation binary cross_entropy at timestep t"""

    def __init__(self, **kwargs):
        super(PermutationLoss, self).__init__()

    def call(self, y_true, y_pred):
        permutation_mixture = y_pred.get('mixture')
        permutation_output = y_pred.get('output')

        batch_size = tf.shape(y_true)[0]
        n_perm = tf.shape(permutation_mixture)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])

        # point wise bce of size : (batch_size, n_perm, T)
        pw_bce = measures.pointwise_bce(y_true_tiled, permutation_output)

        # permutation wise bce of size (batch_size, n_perm)
        permw_bce = tf.math.reduce_sum(pw_bce,
                                       axis=-1)
        tiled_mixture = tf.tile(tf.expand_dims(permutation_mixture, 0),
                                multiples=[batch_size, 1])
        # element wise bce of size (batch_size,) 
        elw_bce = tf.reduce_sum(permw_bce * tiled_mixture, axis=1)
        return tf.reduce_mean(elw_bce)

class T_PermutationLoss(tf.keras.losses.Loss):
    """Compute entropy renormalized permutation loss"""

    def __init__(self, **kwargs):
        super(T_PermutationLoss, self).__init__()
    
    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]

        permutation_mixture = y_pred.get('mixture')
        permutation_output = y_pred.get('output')

        # (n_perm, T)
        permutation_log_T = y_pred.get('log_T')

        # (1, n_perm, T)
        permutation_log_T = tf.expand_dims(permutation_log_T, axis=0)

        # (B, n_perm, T)
        permutation_log_T = tf.tile(permutation_log_T, multiples=[batch_size, 1, 1])
        permutation_T = tf.math.exp(permutation_log_T)

        n_perm = tf.shape(permutation_mixture)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])

        # (batch_size, n_perm, T)
        pw_bce = measures.pointwise_bce(y_true_tiled, permutation_output)

        # (batch_size, n_perm, T)
        perm_wise_loss = (1/2) * permutation_log_T + pw_bce/permutation_T

        # (batch_size, n_perm)
        perm_wise_loss = tf.math.reduce_sum(perm_wise_loss, axis=-1)

        tiled_mixture = tf.tile(tf.expand_dims(permutation_mixture, 0),
                                multiples=[batch_size, 1])
        # element wise bce of size (batch_size,) 
        elw_bce = tf.reduce_sum(perm_wise_loss * tiled_mixture, axis=1)
        return tf.reduce_mean(elw_bce)

class PermutationLossv2(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(PermutationLossv2, self).__init__()

    def call(self, y_true, y_pred):
        permutation_mixture_logits = y_pred.get('mixture_logits')

        permutation_output = y_pred.get('output')

        batch_size = tf.shape(y_true)[0]
        n_perm = tf.shape(permutation_mixture_logits)[0]


        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])

        # (batch_size, n_perm)
        mixture_logits_tiled = tf.tile(tf.expand_dims(permutation_mixture_logits, 0),
                                       multiples=[batch_size, 1])

        # point wise bce of size : (batch_size, n_perm, T)
        pw_bce = measures.pointwise_bce(y_true_tiled, permutation_output)

        # permutation wise bce of size (batch_size, n_perm)
        # L_sigma/batch
        permw_bce = tf.math.reduce_sum(pw_bce,
                                       axis=-1)
        batchw_loss = tf.math.reduce_logsumexp(mixture_logits_tiled, axis=1) - tf.math.reduce_logsumexp(mixture_logits_tiled - permw_bce, axis=1) 

        result = tf.math.reduce_mean(batchw_loss, axis=0)
        return result

class DropoutPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, name='dropout_permutation_loss', **kwargs):
        super(DropoutPermutationLoss, self).__init__(measure=measures.dropout_permutation_loss(pred_in),
                                                     **kwargs)
class DropoutJensenPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, name='dropout_jensen_permutation_loss', **kwargs):
        super(DropoutJensenPermutationLoss, self).__init__(measure=measures.dropout_jensen_permutation_loss(pred_in))

class TreePermutationLoss(MeasureLoss):
    def __init__(self, pred_in, name='tree_loss', **kwargs):
        super(TreePermutationLoss, self).__init__(measure=measures.tree_permutation_loss(pred_in),
                                                  **kwargs)

class TreePermutationLossCategorical(MeasureLoss):
    def __init__(self, pred_in, name='treecat_loss', **kwargs):
        super(TreePermutationLossCategorical, self).__init__(measure=measures.tree_permutation_categorical_loss(pred_in),
                                                             **kwargs)

class BalancedPermutationLoss(tf.keras.losses.Loss):
    def __init__(self, dataset, n_task, **kwargs):
        super(BalancedPermutationLoss, self).__init__(**kwargs)
        self.balanced_pointwise_loss = measures.frequency_balanced_bce(dataset, n_task)

    def call(self, y_true, y_pred):
        permutation_mixture = y_pred.get('mixture')
        permutation_output = y_pred.get('output')

        batch_size = tf.shape(y_true)[0]
        n_perm = tf.shape(permutation_mixture)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])

        # point wise bce of size : (batch_size, n_perm, T)
        pw_bce = self.balanced_pointwise_loss(y_true_tiled, permutation_output)

        # permutation wise bce of size (batch_size, n_perm)
        permw_bce = tf.math.reduce_sum(pw_bce,
                                       axis=-1)
        tiled_mixture = tf.tile(tf.expand_dims(permutation_mixture, 0),
                                multiples=[batch_size, 1])
        # element wise bce of size (batch_size,) 
        elw_bce = tf.reduce_sum(permw_bce * tiled_mixture, axis=1)
        return tf.reduce_mean(elw_bce)

class HierarchicalPermutationLoss(tf.keras.losses.Loss):
    def __init__(self, n_blocks, **kwargs):
        super(HierarchicalPermutationLoss, self).__init__()
        self.n_blocks = n_blocks
    
    def call(self, y_true, y_pred):
        loss_by_block = []
        for k in range(self.n_blocks):
            block_string = 'block_{}'.format(k)
            block = y_pred[block_string]['block']


            # (B, P_1, P_2, B_{k+1})
            block_output = tf.concat(y_pred[block_string]['outputs'], axis=1)
            output_shape = tf.shape(block_output)

            batch_size = output_shape[0]
            p_1 = output_shape[1]
            p_2 = output_shape[2]
            b_k1 = len(block)

            # (B, B_{k+1})
            y_block = tf.reshape(tf.gather(y_true,
                                           block,
                                           axis=-1),
                                 (batch_size, 1, 1, b_k1))

            # (B, P_1, P_2, B_{k+1})
            y_block_tiled = tf.tile(y_block, multiples=[1, p_1, p_2, 1])

            # (B, P_1, P_2, B_{k+1})
            block_pointwise_bce = measures.pointwise_bce(y_block_tiled, block_output)

            # (P_1, P_2)
            block_pointwise_bce = tf.math.reduce_mean(tf.math.reduce_sum(block_pointwise_bce, axis=-1), axis=0)

            # (P_1, P_2)
            block_mixtures = tf.concat(y_pred[block_string]['mixtures'], axis=0)

            # (P_1, )
            blockwise_bce = tf.math.reduce_sum(block_pointwise_bce * block_mixtures, axis=-1)
            loss_by_block.append(tf.expand_dims(blockwise_bce, axis=1))

        # (P_1, N_B)
        concat_loss_block = tf.concat(loss_by_block, axis=1)

        # (P_1 , 1)
        mixture = tf.expand_dims(y_pred['mixture'], axis=-1)
        mixture = tf.tile(mixture, multiples=[1, self.n_blocks])

        return tf.math.reduce_sum(concat_loss_block * mixture)

class SSPCLoss(MeasureLoss):
    def __init__(self, pred_in, name='sspc_loss', **kwargs):
        super(SSPCLoss, self).__init__(measure=measures.semi_supervised_permutation_categorical_loss(pred_in),
                                       **kwargs)

class CategoricalPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, name='cp_loss', **kwargs):
        super(CategoricalPermutationLoss, self).__init__(measure=measures.permutation_categorical_loss(pred_in),
                                                         **kwargs)

class WeightedCategoricalPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, task_weights, name='wcp_loss', **kwargs):
        super(WeightedCategoricalPermutationLoss, self).__init__(measure=measures.weighted_permutation_categorical_loss(pred_in, task_weights))

class Bp4dAUCategoricalPermutationLoss(WeightedCategoricalPermutationLoss):
    def __init__(self, pred_in, occurences, name='bp4d-au_cp_loss', **kwargs):
        n_coords = tf.shape(occurences)[0]
        AU_occurences = tf.reshape(occurences, (n_coords // 2, 2))[:, 0]
        AU_w = (1 / AU_occurences) / tf.math.reduce_sum(1 / AU_occurences)
        super(Bp4dAUCategoricalPermutationLoss, self).__init__(pred_in=pred_in,
                                                               task_weights=AU_w,
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

class WeightedCategoricalPermutationDice(MeasureLoss):
    def __init__(self, pred_in, task_weights, name='wcp_dice', **kwargs):
        super(WeightedCategoricalPermutationDice, self).__init__(measure=measures.weighted_categorical_permutation_dice(pred_in, task_weights))

class WeightedPermutationDice(MeasureLoss):
    def __init__(self, pred_in, task_weights, name='wp_dice', **kwargs):
        super(WeightedPermutationDice, self).__init__(measure=measures.weighted_binary_permutation_dice(pred_in, tf.constant(task_weights, dtype=tf.float32)))

class FrequencyWeightedPermutationDice(WeightedPermutationDice):
    def __init__(self, pred_in, frequencies, name='fwp_dice', **kwargs):
        task_weights = (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FrequencyWeightedPermutationDice, self).__init__(pred_in=pred_in,
                                                               task_weights=task_weights,
                                                               name=name,
                                                               **kwargs)

class SoftFrequencyWeightedPermutationDice(WeightedPermutationDice):
    def __init__(self, pred_in, frequencies, name='sfwp_dice', **kwargs):
        task_weights = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 - frequencies) / tf.math.reduce_sum(1 - frequencies)
        super(SoftFrequencyWeightedPermutationDice, self).__init__(pred_in=pred_in,
                                                                   task_weights=task_weights,
                                                                   name=name,
                                                                   **kwargs)
class Bp4dAUPermutationDice(WeightedCategoricalPermutationDice):
    def __init__(self, pred_in, occurences, name='bp4d-au_dice', **kwargs):
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
class WeightedBinaryPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, task_weights, name='wbp_loss', **kwargs):
        super(WeightedBinaryPermutationLoss, self).__init__(measure=measures.weighted_binary_permutation_loss(pred_in=pred_in,
                                                                                                              task_weights=task_weights,
                                                                                                              **kwargs))

class FrequencyWeightedBinaryPermutationLoss(WeightedBinaryPermutationLoss):
    def __init__(self, pred_in, occurences, name='fwbp_loss', **kwargs):
        task_weights = tf.dtypes.cast(tf.shape(occurences)[0], tf.float32) * (1 / occurences) / tf.math.reduce_sum(1 / occurences)
        super(FrequencyWeightedBinaryPermutationLoss, self).__init__(pred_in=pred_in,
                                                                     task_weights=task_weights,
                                                                     name=name)

class SoftFrequencyWeightedBinaryPermutationLoss(WeightedBinaryPermutationLoss):
    def __init__(self, pred_in, frequencies, name='sfwbp_loss', **kwargs):
        task_weights = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 - frequencies) / tf.math.reduce_sum(1 - frequencies)
        super(SoftFrequencyWeightedBinaryPermutationLoss, self).__init__(pred_in=pred_in,
                                                                         task_weights=task_weights,
                                                                         name=name)

class BalancedBCEPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, weights_positive, weights_negative, name='bbcep_loss', **kwargs):
        super(BalancedBCEPermutationLoss, self).__init__(measure=balanced_bce_permutation_loss(pred_in=pred_in,
                                                                                               weights_positive=weights_positive,
                                                                                               weights_negative=weights_negative),
                                                         **kwargs)

class FrequencyBalancedBCEPermutationLoss(BalancedBCEPermutationLoss):
    def __init__(self, pred_in, frequencies, name='fbbcep_loss', **kwargs):
        weights_positive = 1 - frequencies
        weights_negative = frequencies
        super(FrequencyBalancedBCEPermutationLoss, self).__init__(pred_in=pred_in,
                                                                  weights_positive=weights_positive,
                                                                  weights_negative=weights_negative,
                                                                  **kwargs)

class InverseFrequencyBalancedBCEPermutationLoss(BalancedBCEPermutationLoss):
    def __init__(self, pred_in, frequencies, name='ifbbcep_loss', **kwargs):
        weights_positive = 1 / frequencies
        weights_negative = 1 / (1 - frequencies)
        super(InverseFrequencyBalancedBCEPermutationLoss, self).__init__(pred_in=pred_in,
                                                                         weights_positive=weights_positive,
                                                                         weights_negative=weights_negative,
                                                                         **kwargs)

class WeightedBalancedBCEPermutationLoss(MeasureLoss):
    def __init__(self, pred_in, task_weights, weights_positive, weights_negative, name='wbbcep_loss', **kwargs):
        super(WeightedBalancedBCEPermutationLoss, self).__init__(measure=weighted_balanced_bce_permutation_loss(pred_in=pred_in,
                                                                                                                         weights_positive=weights_positive,
                                                                                                                         weights_negative=weights_negative,
                                                                                                                         task_weights=task_weights),
                                                                                                                         name=name)

class FrequencyWeightedBalancedBCEPermutationLoss(WeightedBalancedBCEPermutationLoss):
    def __init__(self, pred_in, frequencies,  name='fwbbcep_loss', **kwargs):
        weights_positive = 1 - frequencies
        weights_negative = frequencies
        inverse_frequencies = 1 / frequencies
        task_weights = inverse_frequencies / tf.math.reduce_sum(inverse_frequencies)
        super(FrequencyWeightedBalancedBCEPermutationLoss, self).__init__(pred_in=pred_in,
                                                                          weights_positive=weights_positive,
                                                                          weights_negative=weights_negative,
                                                                          task_weights=task_weights,
                                                                          name=name)


SUPPORTED_PERMUTATION_LOSSES = {"balanced_permutation": BalancedPermutationLoss,
                                "permutation": PermutationLoss,
                                "do_perm": DropoutPermutationLoss,
                                "do_jensen_perm": DropoutJensenPermutationLoss,
                                "hpermutation": HierarchicalPermutationLoss,
                                "sspc_loss": SSPCLoss,
                                "tree": TreePermutationLoss,
                                "tree_categorical": TreePermutationLossCategorical,
                                "wcp_loss": WeightedCategoricalPermutationLoss,
                                "wbp_loss": WeightedBinaryPermutationLoss,
                                "fwbp_loss": FrequencyWeightedBinaryPermutationLoss,
                                "sfwbp_loss": SoftFrequencyWeightedBinaryPermutationLoss,
                                "wp_dice": WeightedPermutationDice,
                                "fwp_dice": FrequencyWeightedPermutationDice,
                                "sfwp_dice": SoftFrequencyWeightedPermutationDice,
                                "fbbcep_loss": FrequencyBalancedBCEPermutationLoss,
                                "ifbbcep_loss": InverseFrequencyBalancedBCEPermutationLoss,
                                "fwbbcep_loss": FrequencyWeightedBalancedBCEPermutationLoss,
                                "bp4d-au_cp_loss": Bp4dAUCategoricalPermutationLoss,
                                "bp4d-au_sex_cp_loss": Bp4dAUandSexCategoricalPermutationLoss,
                                "bp4d-au_sex_dice": Bp4dAUandSexPermutationDice,
                                "bp4d-au_dice": Bp4dAUPermutationDice,
                                "T_permutation": T_PermutationLoss}
