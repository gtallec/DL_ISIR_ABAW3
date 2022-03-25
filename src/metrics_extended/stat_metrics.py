from metrics_extended.abstract_metrics import SlidingMeanMetric 
import measures

class Norm(SlidingMeanMetric):
    def __init__(self, pred_in, order, name='norm', **kwargs):
        super(Norm, self).__init__(name=name + '_' + pred_in,
                                   eval_function=measures.mean_norm(pred_in, order))
class Entropy(SlidingMeanMetric):
    def __init__(self, pred_in, name='entropy', **kwargs):
        super(Entropy, self).__init__(name=name + '_' + pred_in,
                                      eval_function=measures.mean_entropy(pred_in))


"""
class EntropyFromLogits(SlidingMeanMetric):
    def __init__(self, mixtures_in, name='entropy', **kwargs):
        super(EntropyFromLogits, self).__init__(name=name + '_' + mixtures_in,
                                                eval_function=measures.mean_entropy(mixtures_in))

"""

SUPPORTED_STAT_METRICS = {"norm": Norm,
                          "entropy": Entropy}
