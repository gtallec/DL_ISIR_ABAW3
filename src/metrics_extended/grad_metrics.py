import tensorflow as tf

from abstract_metrics import ScalarTrackingMetric
import measures

class GradMetric(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(GradMetric, self).__init__()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred['track_grad']
        self.grad_metric(y_pred)

    def grad_metric(self, y_true, y_pred):
        pass


class MeanScalarTransformGradTracker(GradMetric, ScalarTrackingMetric):
    def __init__(self, eval_function, name, **kwargs):
        GradMetric.__init__(self)
        ScalarTrackingMetric.__init__(self, name=name)
        self.eval_function = eval_function

        self.moving_mean = self.add_weight(name=name + '_moving_mean',
                                           initializer='zeros',
                                           dtype=tf.float32)

        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)

    def grad_metric(self, y_true, y_pred):
        """
        y_pred (B, I).
        """
        M = tf.dtypes.cast(tf.shape(y_true)[0], dtype=tf.float32)
        combination_coeff = self.N / (self.N + M)
        batch_eval = self.eval_function(y_pred)
        self.moving_mean.assign(self.moving_mean * combination_coeff + (1 - combination_coeff) * batch_eval)
        self.N.assign_add(M)

    def result(self):
        return self.moving_mean


class GradNorm(MeanScalarTransformGradTracker):
    def __init__(self, pred_in, order):
        super(MeanScalarTransformGradTracker, self).__init(measures.norm(pred_in, order),
                                                           name="gradnorm")


SUPPORTED_GRAD_METRICS = {"gradnorm": GradNorm}


