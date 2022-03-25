import tensorflow as tf

def pointwise_bce(labels, logits):
    zero = tf.constant(0, tf.float32)
    return tf.maximum(logits, zero) - logits * labels + tf.math.log(1 + tf.math.exp(-tf.math.abs(logits)))

def pointwise_balanced_bce(alpha, beta):
    """
    Implements cross entropy giving weight alpha to positive examples and beta to negative examples :
    L(y_true, logits) = - (alpha * y_true * log(sigmoid(logits)) + beta * (1 - y_true) * log(1 - sigmoid(logits))).
    The purpose of this implementation is to make the loss robust to small value of logits and it is therefore
    implemented as :
    L(y_true, logits) = (alpha * y_true + beta * (1 - y_true))(log(1 + exp(-|logits|)) + max(logits, 0)) - alpha * y_true * logits.

    Ins:
    y_true (batchsize, n): Ground truth labels
    logits (batchsize, n): Output of last network layer (before activation function)
    alpha (n,): weights for positive examples.
    beta (n,): weights for negative examples.

    Out:
    L (batchsize, n) : Mean by examples by coordinate
    """
    def loss(y_true, logits):
        zero = tf.constant(0, tf.float32)
        y_true = tf.dtypes.cast(y_true, dtype=tf.float32)
        logits = tf.dtypes.cast(logits, dtype=tf.float32)
        loss_by_examples_by_coord = ((alpha * y_true + beta * (1 - y_true)) *
                                     (tf.math.log(1 + tf.math.exp(-tf.math.abs(logits))) + tf.math.maximum(logits, zero)) -
                                     alpha * y_true * logits)
        return loss_by_examples_by_coord
    return loss

def weighted_bce(pred_in, weights):
    def fun(y_true, y_pred):
        T = tf.shape(y_true)[-1]
        y_true = tf.reshape(y_true, (-1, T))
        # (B, T)
        logits = y_pred[pred_in]
        logits = tf.reshape(logits, (-1, T))
        # (B, T)
        mlog_p = pointwise_bce(y_true, logits)
        # (B, )
        w_mlog_p = tf.math.reduce_sum(mlog_p * weights[tf.newaxis, :], axis=1)
        return tf.math.reduce_mean(w_mlog_p)
    return fun

def uncertainty_weighted_bce(pred_in, weights_in):
    def fun(y_true, y_pred):
        # (B, T)
        logits = y_pred[pred_in]
        log_u = y_pred[weights_in]
        inv_u2 = tf.math.exp(-2 * log_u)

        mlog_p = pointwise_bce(y_true, logits)
        w_mlog_p = tf.math.reduce_sum(mlog_p * inv_u2 + log_u, axis=1)
        loss = tf.math.reduce_mean(w_mlog_p)
        return loss
    return fun
        
def weighted_binary_permutation_loss(pred_in, task_weights):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        # (B, k)
        pi_im = y_pred.get('mixture')
        # (B, k, T)
        o_imt = y_pred.get('output')

        pi_im_shape = tf.shape(pi_im)
        B = pi_im_shape[0]
        k = pi_im_shape[1]
        T = tf.shape(task_weights)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, k, 1])

        # (B, k, T)
        log_p_sigma_imt = - pointwise_bce(y_true_tiled, o_imt)
        # (B, k, T)
        tiled_task_weights = tf.tile(tf.reshape(task_weights, (1, 1, T)), (B, k, 1))

        # (B, k)
        log_p_sigma_im = tf.math.reduce_sum(log_p_sigma_imt * tiled_task_weights, axis=-1)

        # (B,) 
        L_i = - tf.math.reduce_sum(log_p_sigma_im * pi_im, axis=1)
        return tf.math.reduce_sum(L_i)
    return fun

def balanced_bce_permutation_loss(pred_in, weights_positive, weights_negative):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]

        # (B, k)
        pi_im = y_pred.get('mixture')
        # (B, k, T)
        o_imt = y_pred.get('prediction_logits')

        pi_im_shape = tf.shape(pi_im)
        B = pi_im_shape[0]
        k = pi_im_shape[1]
        T = tf.shape(weights_negative)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, k, 1])

        # (B, k, T)
        weighted_bce_imt = pointwise_balanced_bce(weights_positive, weights_negative)(y_true_tiled, o_imt)
        # (T,)
        negative_ponderation = 1 / weights_negative
        negative_ponderation = tf.tile(tf.reshape(negative_ponderation, (1, 1, T)), (B, k, 1))

        # (B, k, T)

        # (B, k)
        permwise_bce = tf.math.reduce_sum(weighted_bce_imt * negative_ponderation, axis=-1)

        # (B,) 
        L_i = tf.math.reduce_sum(permwise_bce * pi_im, axis=1)
        return tf.math.reduce_mean(L_i)
    return fun

def weighted_balanced_bce_permutation_loss(pred_in, task_weights, weights_positive, weights_negative): 
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]

        # (B, k)
        pi_im = y_pred.get('mixture')
        # (B, k, T)
        o_imt = y_pred.get('prediction_logits')

        pi_im_shape = tf.shape(pi_im)
        B = pi_im_shape[0]
        k = pi_im_shape[1]
        T = tf.shape(weights_negative)[0]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, k, 1])

        # (B, k, T)
        weighted_bce_imt = pointwise_balanced_bce(weights_positive, weights_negative)(y_true_tiled, o_imt)
        # (T,)
        negative_ponderation = 1 / weights_negative
        negative_ponderation = tf.tile(tf.reshape(negative_ponderation, (1, 1, T)), (B, k, 1))

        task_ponderation = tf.tile(tf.reshape(task_weights, (1, 1, T)), (B, k, 1))
        ponderation = negative_ponderation * task_ponderation

        # (B, k, T)

        # (B, k)
        permwise_bce = tf.math.reduce_sum(weighted_bce_imt * ponderation, axis=-1)

        # (B,) 
        L_i = tf.math.reduce_sum(permwise_bce * pi_im, axis=1)
        return tf.math.reduce_mean(L_i)
    return fun

def weighted_dice(pred_in, weights, epsilon=1.0):
    epsilon = tf.constant(epsilon, dtype=tf.float32)
    
    def fun(y_true, y_pred):
        # (B, T)
        logits = y_pred[pred_in]
        predictions = tf.nn.sigmoid(logits)
        coordwise_dice = 1 - (2 * y_true * predictions + epsilon) / (tf.math.square(y_true)
                                                                     + tf.math.square(predictions)
                                                                     + epsilon)
        return tf.math.reduce_mean(tf.math.reduce_sum(coordwise_dice * weights[tf.newaxis, :], axis=-1))
    return fun


