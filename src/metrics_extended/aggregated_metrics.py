import tensorflow as tf

class MeanMetric(tf.keras.metrics.Metric):
    def __init__(self, metric, **kwargs):
        (super(MeanMetric, self)
         .__init__(**kwargs))
        self.metric_to_mean = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.metric_to_mean.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return tf.math.reduce_mean(self.metric.result())

    def reset_states(self):
        self.metric_to_mean.reset_states()


        
