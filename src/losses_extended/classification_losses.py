import tensorflow as tf
import measures
import numpy as np

from losses_extended.abstract_losses import MeasureLoss
from measures_extended.reweighted_measures import weighted_bce, uncertainty_weighted_bce
from measures_extended.partial_labels_measures import partial_labels_bce, partial_labels_dice, partial_labels_weighted_bce, partial_labels_weighted_dice

class BTM_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(BTM_Loss, self).__init__()
        self.measure = measures.btm_cce(**kwargs)

    def call(self, y_true, y_pred):
        return self.measure(y_true, y_pred)

class WeightedCrossentropy(tf.keras.losses.Loss):
    def __init__(self, alpha, beta, name='weighted_crossentropy', **kwargs):
        (super(WeightedCrossentropy, self)
         .__init__(name=name, **kwargs))
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        return measures.weighted_crossentropy_from_logits(y_true, y_pred, self.alpha, self.beta)

class BalancedCrossentropy(WeightedCrossentropy):
    def __init__(self, balance, name='balanced_crossentropy', **kwargs):
        (super(BalancedCrossentropy, self)
         .__init__(alpha=balance, beta=1-balance, name=name, **kwargs))

class SemiBCE(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SemiBCE, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        # W_t : (T, )
        eps = 1e-6
        W_t = y_pred['task_mixture']
        norm_W_t = measures.l2_norm(W_t)
        U_t = tf.expand_dims(W_t / norm_W_t, axis=0)

        T = tf.shape(U_t)[0]
        batch_size = tf.shape(y_true)[0]

        # pi_t : (batch_size, T)
        pi_t = y_pred['pi']

        # B_t : (T, )
        B_t = y_pred['soft_order']

        # B_t_tiled : (batch_size, T)
        B_t_tiled = tf.tile(tf.expand_dims(B_t, 0),
                            [batch_size, 1])



        # tiled_U_t : (batch_size, T)
        tiled_U_t = tf.tile(U_t,
                            [batch_size, 1])
        tiled_square_U_t = tf.math.square(tiled_U_t)


        # tiled_mixture : (batch_size, T)
        tiled_mixture = tf.tile(tf.matmul(y_true, tf.transpose(U_t, perm=[1, 0])),
                                [1, T])

        # pos_gaussian : (batch_size, T)
        variance = 1 - tiled_square_U_t + eps
        pos_gaussian = tf.math.exp(- (1/2) * tf.math.square(tiled_mixture - tiled_U_t)/variance)

        # neg_gaussian : (batch_size, T)
        neg_gaussian = tf.math.exp(- (1/2) * tf.math.square(tiled_mixture)/variance)

        # normalisation : (batch_size, T)
        normalisation = 1 / (tf.math.sqrt(2 * np.pi * variance) * norm_W_t)

        # gaussian_mixture : (batch_size, T)
        gaussian_mixture = normalisation * (pi_t * (pos_gaussian) + (1 - pi_t) * neg_gaussian)
        log_gaussian_mixture = tf.math.log(gaussian_mixture)
        loss = - tf.math.reduce_mean(tf.math.reduce_sum(B_t_tiled * log_gaussian_mixture, axis=1))

        return loss

class BinaryCrossentropy(MeasureLoss):
    def __init__(self, pred_in, name='bce', **kwargs):
        (super(BinaryCrossentropy, self)
         .__init__(measure=measures.mean_bce(pred_in),
                   name=name,
                   **kwargs))

class MBinaryCrossentropy(MeasureLoss):
    def __init__(self, pred_in, name='mbce', **kwargs):
        (super(MBinaryCrossentropy, self)
         .__init__(measure=measures.masked_mean_bce(pred_in),
                   name=name,
                   **kwargs))

class WeightedBinaryCrossentropy(MeasureLoss):
    def __init__(self, pred_in, weights, name='wbce', **kwargs):
        super(WeightedBinaryCrossentropy, self).__init__(measure=weighted_bce(pred_in=pred_in,
                                                                              weights=weights),
                                                         name=name,
                                                         **kwargs)

class SoftFrequencyWeightedBinaryCrossentropy(WeightedBinaryCrossentropy):
    def __init__(self, pred_in, frequencies, name='sfwbce', **kwargs):
        weights = (tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 - frequencies) / tf.math.reduce_sum(1 - frequencies))
        super(SoftFrequencyWeightedBinaryCrossentropy, self).__init__(pred_in=pred_in,
                                                                      weights=weights,
                                                                      name=name)

class FrequencyWeightedBinaryCrossentropy(WeightedBinaryCrossentropy):
    def __init__(self, pred_in, frequencies, name='fwbce', **kwargs):
        weights = (tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies))
        super(FrequencyWeightedBinaryCrossentropy, self).__init__(pred_in=pred_in,
                                                                  weights=weights,
                                                                  name=name)

class UncertaintyWeightedBinaryCrossentropy(MeasureLoss):
    def __init__(self, pred_in, weights_in, name='ubce', **kwargs):
        super(UncertaintyWeightedBinaryCrossentropy, self).__init__(measure=uncertainty_weighted_bce(pred_in=pred_in,
                                                                                                     weights_in=weights_in),
                                                                    name=name,
                                                                    **kwargs)

class PLBinaryCrossentropy(MeasureLoss):
    def __init__(self, pred_in, name='plbce', **kwargs):
        super(PLBinaryCrossentropy, self).__init__(measure=partial_labels_bce(pred_in),
                                                   name=name,
                                                   **kwargs)

class PLWBCE(MeasureLoss):
    def __init__(self, pred_in, weights, name='plwbce', **kwargs):
        super(PLWBCE, self).__init__(measure=partial_labels_weighted_bce(pred_in, weights),
                                     name=name,
                                     **kwargs)

class PLFWBCE(PLWBCE):
    def __init__(self, pred_in, plfrequencies, name='plfwbce', **kwargs):
        T = tf.dtypes.cast(tf.shape(plfrequencies)[0], tf.float32)
        mask = 1 - tf.dtypes.cast(plfrequencies == 0, dtype=tf.float32)
        masked_pl_frequencies = (1 / (plfrequencies + 1e-7)) * mask
        masked_weights = T * masked_pl_frequencies / tf.math.reduce_sum(masked_pl_frequencies)
        super(PLFWBCE, self).__init__(pred_in=pred_in,
                                      weights=masked_weights,
                                      name=name)

class PLDice(MeasureLoss):
    def __init__(self, pred_in, name='pldice', **kwargs):
        super(PLDice, self).__init__(measure=partial_labels_dice(pred_in),
                                     name=name,
                                     **kwargs)

class PLWDice(MeasureLoss):
    def __init__(self, pred_in, weights, name='plwdice', **kwargs):
        super(PLWDice, self).__init__(measure=partial_labels_weighted_dice(pred_in, weights=weights),
                                      name=name)

class PLFWDice(PLWDice):
    def __init__(self, pred_in, plfrequencies, name='plfwdice', **kwargs):
        T = tf.dtypes.cast(tf.shape(plfrequencies)[0], tf.float32)
        mask = 1 - tf.dtypes.cast(plfrequencies == 0, dtype=tf.float32)
        masked_pl_frequencies = (1 / (plfrequencies + 1e-7)) * mask
        masked_weights = T * masked_pl_frequencies / tf.math.reduce_sum(masked_pl_frequencies)
        super(PLFWDice, self).__init__(pred_in=pred_in,
                                       weights=masked_weights,
                                       name=name)


SUPPORTED_CLASSIFICATION_LOSSES = {"bce": BinaryCrossentropy,
                                   "mbce": MBinaryCrossentropy,
                                   "plbce": PLBinaryCrossentropy,
                                   "pldice": PLDice,
                                   "plfwbce": PLFWBCE,
                                   "plfwdice": PLFWDice,
                                   "fwbce": FrequencyWeightedBinaryCrossentropy,
                                   "sfwbce": SoftFrequencyWeightedBinaryCrossentropy,
                                   "ubce": UncertaintyWeightedBinaryCrossentropy}
