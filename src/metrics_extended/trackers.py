import tensorflow as tf

class ScalarTracker(tf.keras.metrics.Metric):
    def __init__(self, name="scalar_tracker", **kwargs):
        super(ScalarTracker, self).__init__(name=name)
        self.scalar = self.add_weight(name='scalar',
                                      initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.scalar.assign(y_pred)

    def result(self):
        return self.scalar

class MeanTracker(tf.keras.metrics.Metric):
    def __init__(self, name="mean_tracker", **kwargs):
        super(MeanTracker, self).__init__(name=name)
        self.mean = self.add_weight(name='mean',
                                    initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean.assign(tf.math.reduce_mean(y_pred))

    def result(self):
        return self.mean



SUPPORTED_TRACKERS = {"scalar": ScalarTracker,
                      "mean": MeanTracker}

if __name__ == '__main__':

    scalar_tracker = ScalarTracker()
    scalar_tracker.update_state(y_true=1, y_pred=5)
    print(scalar_tracker.result())

    mean_tracker = MeanTracker()
    mean_tracker.update_state(y_true=1, y_pred=tf.ones((5,)))
    print(mean_tracker.result())
