import tensorflow as tf
from measures import pointwise_bce

def partial_labels_bce(pred_in):
    def fun(y_true, y_pred):
        """
        labels: (..., T)
        logits: (..., T)
        """
        T = tf.shape(y_true)[-1]
        y_pred = y_pred[pred_in]
        y_true = tf.reshape(y_true, (-1, T))
        # (B, T)
        partial_label_mask = 1 - tf.dtypes.cast(y_true == -1, dtype=tf.float32)
        y_true = y_true * partial_label_mask

        y_pred = tf.reshape(y_pred, (-1, T))
        # (B, T)
        tw_bce = pointwise_bce(y_true, y_pred) * partial_label_mask
        # (B, )
        return tf.math.reduce_mean(tw_bce)
    return fun

def partial_labels_weighted_bce(pred_in, weights):
    """ weights (T, ) """
    weights = tf.constant(weights, dtype=tf.float32)

    def fun(y_true, y_pred):
        """
        labels: (..., T)
        logits: (..., T)
        """
        T = tf.shape(y_true)[-1]
        y_pred = y_pred[pred_in]
        y_true = tf.reshape(y_true, (-1, T))
        # (B, T)
        partial_label_mask = 1 - tf.dtypes.cast(y_true == -1, dtype=tf.float32)
        y_true = y_true * partial_label_mask
        y_pred = tf.reshape(y_pred, (-1, T))
        # (B, T)
        tw_bce = pointwise_bce(y_true, y_pred) * partial_label_mask * weights[tf.newaxis, :]
        # (B, )
        return tf.math.reduce_mean(tw_bce)
    return fun

def partial_labels_dice(pred_in, epsilon=1.0):
    epsilon = tf.constant(epsilon, dtype=tf.float32)
    
    def fun(y_true, y_pred):
        # (B, T)
        logits = y_pred[pred_in]
        # (B, T)
        predictions = tf.nn.sigmoid(logits)
        # (B, T)
        mask = 1 - tf.dtypes.cast(y_true == -1, dtype=tf.float32)
        coordwise_dice = 1 - (2 * y_true * predictions + epsilon) / (tf.math.square(y_true)
                                                                     + tf.math.square(predictions)
                                                                     + epsilon)
        return tf.math.reduce_mean(mask * coordwise_dice)
    return fun

def partial_labels_weighted_dice(pred_in, weights, epsilon=1.0):
    weights = tf.constant(weights, dtype=tf.float32)
    epsilon = tf.constant(epsilon, dtype=tf.float32)
    
    def fun(y_true, y_pred):
        # (B, T)
        logits = y_pred[pred_in]
        # (B, T)
        predictions = tf.nn.sigmoid(logits)
        # (B, T)
        mask = 1 - tf.dtypes.cast(y_true == -1, dtype=tf.float32)
        coordwise_dice = 1 - (2 * y_true * predictions + epsilon) / (tf.math.square(y_true)
                                                                     + tf.math.square(predictions)
                                                                     + epsilon)
        return tf.math.reduce_mean(mask * coordwise_dice * weights[tf.newaxis, :])
    return fun
