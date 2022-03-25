import tensorflow as tf
from measures_extended.reweighted_measures import weighted_bce, weighted_dice
from losses_extended.abstract_losses import MeasureLoss


class WeightedBCE(MeasureLoss):
    def __init__(self, pred_in, weights, name='wbce', **kwargs):
        super(WeightedBCE, self).__init__(measure=weighted_bce(pred_in=pred_in,
                                                               weights=weights,
                                                               **kwargs),
                                          name=name)
class FrequencyWeightedBCE(WeightedBCE):
    def __init__(self, pred_in, frequencies, name='fwbce', **kwargs):
        T = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32)
        weights = T * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FrequencyWeightedBCE, self).__init__(pred_in=pred_in,
                                                   weights=weights,
                                                   name=name)

class WeightedDice(MeasureLoss):
    def __init__(self, pred_in, weights, epsilon=1.0, name='wdice', **kwargs):
        super(WeightedDice, self).__init__(measure=weighted_dice(pred_in=pred_in,
                                                                 weights=weights,
                                                                 epsilon=epsilon,
                                                                 **kwargs),
                                           name=name)

class FrequencyWeightedDice(WeightedDice):
    def __init__(self, pred_in, frequencies, epsilon=1.0, name='fwdice', **kwargs):
        T = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32)
        weights = T * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FrequencyWeightedDice, self).__init__(pred_in=pred_in,
                                                    weights=weights,
                                                    epsilon=epsilon,
                                                    name=name)


SUPPORTED_REWEIGHTED_LOSSES = {"fwbce": FrequencyWeightedBCE,
                               "fwdice": FrequencyWeightedDice}




        

