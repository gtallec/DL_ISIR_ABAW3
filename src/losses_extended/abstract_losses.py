import tensorflow as tf

# from loss_builder import get_loss

"""
class SubCoordLoss(tf.keras.losses.Loss):
    def __init__(self, loss_dict, subcoords, **kwargs):
        super(SubCoordLoss, self).__init__()
        self.loss = get_loss(loss_dict)
        self.subcoords = subcoords

    def call(self, y_true, y_pred):
        return self.loss(tf.gather(y_true, self.subcoords, axis=1), y_pred)
"""

class MeasureLoss(tf.keras.losses.Loss):
    def __init__(self, measure, **kwargs):
        super(MeasureLoss, self).__init__()
        self.measure = measure

    def call(self, y_true, y_pred):
        return self.measure(y_true, y_pred)
