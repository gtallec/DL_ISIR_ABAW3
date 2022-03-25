import tensorflow as tf
import numpy as np

import reducers
import mappers
import utils

import itertools


def pointwise_bce(labels, logits):
    zero = tf.constant(0, tf.float32)
    return tf.maximum(logits, zero) - logits * labels + tf.math.log(1 + tf.math.exp(-tf.math.abs(logits)))

def mean_bce(pred_in):
    def fun(y_true, y_pred):
        """
        labels: (..., T)
        logits: (..., T)
        """
        T = tf.shape(y_true)[-1]
        y_pred = y_pred[pred_in]
        y_true = tf.reshape(y_true, (-1, T))
        y_pred = tf.reshape(y_pred, (-1, T))
        # (B, T)
        tw_bce = pointwise_bce(y_true, y_pred)
        # (B, )
        bce = tf.math.reduce_sum(tw_bce, axis=-1)
        return tf.math.reduce_mean(bce)
    return fun

if __name__ == '__main__':
    pass




