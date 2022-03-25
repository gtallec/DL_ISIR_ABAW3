import measures

import tensorflow as tf

from metrics_extended.abstract_metrics import SlidingMeanMetric
from measures_extended.reweighted_measures import weighted_bce
from measures_extended.partial_labels_measures import partial_labels_bce, partial_labels_dice, partial_labels_weighted_bce, partial_labels_weighted_dice

class BCE(SlidingMeanMetric):
    def __init__(self, pred_in, name='bce', **kwargs):
        (super(BCE, self).__init__(name=name,
                                   eval_function=measures.mean_bce(pred_in)))
class PLBCE(SlidingMeanMetric):
    def __init__(self, pred_in, name='plbce', **kwargs):
        (super(PLBCE, self).__init__(name=name,
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

class WBCE(SlidingMeanMetric):
    def __init__(self, pred_in, weights, name='wbce', **kwargs):
        super(WBCE, self).__init__(eval_function=weighted_bce(pred_in=pred_in,
                                                              weights=weights),
                                   name=name)

class FWBCE(WBCE):
    def __init__(self, pred_in, frequencies, name='fwbce', **kwargs):
        weights = tf.dtypes.cast(tf.shape(frequencies)[0], tf.float32) * (1 / frequencies) / tf.math.reduce_sum(1 / frequencies)
        super(FWBCE, self).__init__(pred_in=pred_in,
                                    weights=weights,
                                    name=name)


SUPPORTED_LOSS_METRICS = {"bce": BCE,
                          "plbce": PLBCE,
                          "plfwbce": PLFWBCE,
                          "pldice": PLDice,
                          "plfwdice": PLFWDice,
                          "fwbce": FWBCE}
