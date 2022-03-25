import tensorflow as tf

class MeasureLoss(tf.keras.losses.Loss):
    def __init__(self, measure, **kwargs):
        super(MeasureLoss, self).__init__()
        self.measure = measure

    def call(self, y_true, y_pred):
        return self.measure(y_true, y_pred)
