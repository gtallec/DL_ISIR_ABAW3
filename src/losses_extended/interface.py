import tensorflow as tf

from losses_extended.loss_builder import get_loss

class MultiOutputLosses(tf.keras.losses.Loss):
    def __init__(self, loss_kwargs_list, **kwargs):
        super(MultiOutputLosses, self).__init__(**kwargs)
        self.loss_list = []
        self.weights = []

        for i in range(len(loss_kwargs_list)):
            loss_kwargs = loss_kwargs_list[i]
            weight = loss_kwargs.pop('weight', 1.0)

            self.loss_list.append(get_loss(loss_kwargs))
            self.weights.append(weight)

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum([self.weights[i] * self.loss_list[i](y_true, y_pred) for i in range(len(self.loss_list))])


def get_losses(losses_dict):
    return MultiOutputLosses(losses_dict)
