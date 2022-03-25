import tensorflow as tf
import measures

from losses_extended.abstract_losses import MeasureLoss
from measures_extended.reweighted_measures import weighted_bce, weighted_dice 
from measures_extended.partial_labels_measures import partial_labels_bce, partial_labels_dice, partial_labels_weighted_bce, partial_labels_weighted_dice

class BCE(MeasureLoss):
    def __init__(self, pred_in, name='bce', **kwargs):
        (super(BCE, self)
         .__init__(measure=measures.mean_bce(pred_in),
                   name=name,
                   **kwargs))

class WBCE(MeasureLoss):
    def __init__(self, pred_in, weights, name='wbce', **kwargs):
        super(WBCE, self).__init__(measure=weighted_bce(pred_in=pred_in,
                                                        weights=weights),
                                   name=name,
                                   **kwargs)

class FWBCE(WBCE):
    def __init__(self, pred_in, frequencies, name='fwbce', **kwargs):
        weights = (tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies))
        super(FWBCE, self).__init__(pred_in=pred_in,
                                    weights=weights,
                                    name=name)

class PLBCE(MeasureLoss):
    def __init__(self, pred_in, name='plbce', **kwargs):
        super(PLBCE, self).__init__(measure=partial_labels_bce(pred_in),
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


class WDice(MeasureLoss):
    def __init__(self, pred_in, weights, epsilon=1.0, name='wdice', **kwargs):
        super(WDice, self).__init__(measure=weighted_dice(pred_in=pred_in,
                                                          weights=weights,
                                                          epsilon=epsilon,
                                                          **kwargs),
                                    name=name)

class FWDice(WDice):
    def __init__(self, pred_in, frequencies, epsilon=1.0, name='fwdice', **kwargs):
        T = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32)
        weights = T * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FWDice, self).__init__(pred_in=pred_in,
                                     weights=weights,
                                     epsilon=epsilon,
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


SUPPORTED_CLASSIFICATION_LOSSES = {"bce": BCE,
                                   "fwbce": FWBCE,
                                   "plbce": PLBCE,
                                   "plfwbce": PLFWBCE,
                                   "pldice": PLDice,
                                   "fwdice": FWDice,
                                   "plfwdice": PLFWDice}
