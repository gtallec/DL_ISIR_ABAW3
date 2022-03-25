from metrics_extended.abstract_metrics import ParameterTrackingMetric
import tensorflow as tf

def norm(order):
    def fun(param):
        return tf.norm(param, ord=order)
    return fun

def standard_deviation(param):
    return tf.math.reduce_std(param)

class ParameterNorm(ParameterTrackingMetric):
    def __init__(self, param_in, order, name='norm', **kwargs):
        super(ParameterNorm, self).__init__(name=name,
                                            param_in=param_in,
                                            eval_function=norm(order))

class ParameterStandardDeviation(ParameterTrackingMetric):
    def __init__(self, param_in, name='var', **kwargs):
        super(ParameterStandardDeviation, self).__init__(name=name,
                                                         param_in=param_in,
                                                         eval_function=standard_deviation)


SUPPORTED_PARAMETER_TRACKERS = {"param_norm": ParameterNorm,
                                "param_std": ParameterStandardDeviation}
