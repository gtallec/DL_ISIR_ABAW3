import tensorflow as tf
from metrics_extended.abstract_metrics import SlidingMeanMetric
from measures_extended.reweighted_measures import weighted_bce, weighted_dice


class WeightedBCE(SlidingMeanMetric):
    def __init__(self, pred_in, weights, name="wbce", **kwargs):
        (super(WeightedBCE, self)
         .__init__(eval_function=weighted_bce(pred_in, weights),
                   name=name))

class FrequencyWeightedBCE(WeightedBCE):
    def __init__(self, pred_in, frequencies, name="fwbce", **kwargs):
        T = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32)
        weights = T * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        (super(FrequencyWeightedBCE, self)
         .__init__(pred_in=pred_in,
                   weights=weights,
                   name=name))

class WeightedDice(SlidingMeanMetric):
    def __init__(self, pred_in, weights, epsilon=1.0, name="wdice", **kwargs):
        (super(WeightedDice, self)
         .__init__(eval_function=weighted_dice(pred_in, weights, epsilon=epsilon),
                   name=name))

class FrequencyWeightedDice(WeightedDice):
    def __init__(self, pred_in, frequencies, epsilon=1.0, name="fwdice", **kwargs):
        T = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32)
        weights = T * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        (super(FrequencyWeightedDice, self)
         .__init__(pred_in=pred_in,
                   weights=weights,
                   epsilon=epsilon,
                   name=name))


SUPPORTED_REWEIGHTED_METRICS = {"fwbce": FrequencyWeightedBCE,
                                "fwdice": FrequencyWeightedDice}
