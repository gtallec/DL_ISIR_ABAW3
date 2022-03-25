import measures

import tensorflow as tf

from metrics_extended.abstract_metrics import SlidingMeanMetric
from measures_extended.reweighted_measures import weighted_bce, uncertainty_weighted_bce
from measures_extended.partial_labels_measures import partial_labels_bce, partial_labels_dice, partial_labels_weighted_bce, partial_labels_weighted_dice


class BTMCCE(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(BTMCCE, self)
         .__init__(name="BTMCCE",
                   eval_function=measures.btm_cce(**kwargs)))

class BinaryCrossentropy(SlidingMeanMetric):
    def __init__(self, pred_in, name='bce', **kwargs):
        (super(BinaryCrossentropy, self).__init__(name=name,
                                                  eval_function=measures.mean_bce(pred_in)))

class MBinaryCrossentropy(SlidingMeanMetric):
    def __init__(self, pred_in, name='mbce', **kwargs):
        (super(MBinaryCrossentropy, self).__init__(name=name,
                                                   eval_function=measures.masked_mean_bce(pred_in)))

class PLBinaryCrossentropy(SlidingMeanMetric):
    def __init__(self, pred_in, name='plbce', **kwargs):
        (super(PLBinaryCrossentropy, self).__init__(name=name,
                                                    eval_function=partial_labels_bce(pred_in)))

class PLDice(SlidingMeanMetric):
    def __init__(self, pred_in, name='pldice', **kwargs):
        (super(PLDice, self).__init__(name=name,
                                      eval_function=partial_labels_dice(pred_in)))
class PLWDice(SlidingMeanMetric):
    def __init__(self, pred_in, weights, name='plwdice', **kwargs):
        (super(PLWDice, self).__init__(name=name,
                                       eval_function=partial_labels_weighted_dice(pred_in, weights=weights)))

class PLWBCE(SlidingMeanMetric):
    def __init__(self, pred_in, weights, name='plwbce', **kwargs):
        (super(PLWBCE, self).__init__(name=name,
                                      eval_function=partial_labels_weighted_bce(pred_in=pred_in,
                                                                                weights=weights)))

class PLFWBCE(PLWBCE):
    def __init__(self, pred_in, plfrequencies, name='plfwbce', **kwargs):
        weights = tf.dtypes.cast(tf.shape(plfrequencies)[0], tf.float32) * (1 / plfrequencies) / tf.math.reduce_sum(1 / plfrequencies)
        super(PLFWBCE, self).__init__(pred_in=pred_in,
                                      weights=weights,
                                      name=name)
class PLFWDice(PLWDice):
    def __init__(self, pred_in, plfrequencies, name='plfwdice', **kwargs):
        weights = tf.dtypes.cast(tf.shape(plfrequencies)[0], tf.float32) * (1 / plfrequencies) / tf.math.reduce_sum(1 / plfrequencies)
        super(PLFWDice, self).__init__(pred_in=pred_in,
                                       weights=weights,
                                       name=name)

class PositiveBCE(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(PositiveBCE, self)
         .__init__(name='positive_bce',
                   eval_function=measures.positivelabels_bce))

class NegativeBCE(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(NegativeBCE, self)
         .__init__(name='negative_bce',
                   eval_function=measures.negativelabels_bce))

class WeightedBinaryCrossentropy(SlidingMeanMetric):
    def __init__(self, pred_in, weights, name='wbce', **kwargs):
        super(WeightedBinaryCrossentropy, self).__init__(eval_function=weighted_bce(pred_in=pred_in,
                                                                                    weights=weights),
                                                         name=name)

class FrequencyWeightedBinaryCrossentropy(WeightedBinaryCrossentropy):
    def __init__(self, pred_in, frequencies, name='fwbce', **kwargs):
        weights = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FrequencyWeightedBinaryCrossentropy, self).__init__(pred_in=pred_in,
                                                                  weights=weights,
                                                                  name=name)


class SoftFrequencyWeightedBinaryCrossentropy(WeightedBinaryCrossentropy):
    def __init__(self, pred_in, frequencies, name='sfwbce', **kwargs):
        weights = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 - frequencies) / tf.math.reduce_sum(1 - frequencies)
        super(SoftFrequencyWeightedBinaryCrossentropy, self).__init__(pred_in=pred_in,
                                                                      weights=weights,
                                                                      name=name)

class UncertaintyWeightedBinaryCrossentropy(SlidingMeanMetric):
    def __init__(self, pred_in, weights_in, name='ubce', **kwargs):
        super(UncertaintyWeightedBinaryCrossentropy, self).__init__(eval_function=uncertainty_weighted_bce(pred_in=pred_in,
                                                                                                           weights_in=weights_in),
                                                                    name=name,
                                                                    **kwargs)


SUPPORTED_LOSS_METRICS = {"bce": BinaryCrossentropy,
                          "mbce": MBinaryCrossentropy,
                          "plbce": PLBinaryCrossentropy,
                          "plfwbce": PLFWBCE,
                          "pldice": PLDice,
                          "plfwdice": PLFWDice,
                          "ubce": UncertaintyWeightedBinaryCrossentropy,
                          "sfwbce": SoftFrequencyWeightedBinaryCrossentropy,
                          "fwbce": FrequencyWeightedBinaryCrossentropy,
                          "pbce": PositiveBCE,
                          "nbce": NegativeBCE,
                          "mse": tf.keras.metrics.MeanSquaredError,
                          "cce": tf.keras.metrics.CategoricalCrossentropy,
                          "btm_cce": BTMCCE}
