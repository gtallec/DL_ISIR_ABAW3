import tensorflow as tf
import measures
from metrics_extended.abstract_metrics import SlidingMeanMetric


class KL_div_to_uniform(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(KL_div_to_uniform, self)
         .__init__(name='kl2uni',
                   eval_function=measures.KL_divergence))


class Argmax(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        (super(Argmax, self)
         .__init__(**kwargs))
        self.main_id = self.add_weight(name='main_id',
                                       initializer='zeros',
                                       dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.main_id.assign(tf.math.argmax(y_pred))

    def result(self):
        return self.main_id

class Norm(tf.keras.metrics.Metric):
    def __init__(self, order, **kwargs):
        (super(Norm, self)
         .__init__(**kwargs))

        self.l2_norm = self.add_weight(name='main_id',
                                       initializer='zeros',
                                       dtype=tf.float32)
        self.order = order

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.l2_norm.assign(tf.norm(y_pred, ord=self.order))

    def result(self):
        return self.l2_norm


SUPPORTED_UTILS_METRICS = {"argmax": Argmax,
                           "kl2uni": KL_div_to_uniform,
                           "norm": Norm}
