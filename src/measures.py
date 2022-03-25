import tensorflow as tf
import numpy as np

import reducers
import mappers
import utils

import itertools


def mean_batch(y_true, y_pred):
    return tf.math.reduce_mean(y_pred)

def std_batch(y_true, y_pred):
    return tf.math.reduce_std(y_pred)

def task_permutation_losses(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]

        # (B, k)
        pi_im = y_pred.get('mixture')
        # (B, k, T)
        o_imt = y_pred.get('prediction_logits')

        k = tf.shape(pi_im)[1]
        T = tf.shape(o_imt)[-1]

        pi_imt = tf.tile(tf.expand_dims(pi_im, axis=-1), (1, 1, T)) 
        # (B, k, T)
        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, k, 1])

        # (B, k, T)
        log_p_sigma_imt = - pointwise_bce(y_true_tiled, o_imt)
        # (B, T)
        log_p_sigma_it = tf.math.reduce_sum(log_p_sigma_imt * pi_imt, axis=1)
        # (T, )
        log_p_sigma_t = tf.math.reduce_mean(log_p_sigma_it, axis=0)
        return - log_p_sigma_t
    return fun

def distance_matrix_fun(vectors_in):
    def fun(y_true, y_pred):
        # (B, T, F)
        tokens = y_pred[vectors_in]
        return distance_matrix(tokens)
    return fun

def distance_matrix(tokens):
    # (B, T)
    token_norms = tf.math.reduce_sum(tf.math.pow(tokens, 2), axis=-1)
    tf.print(token_norms.shape)
    return (token_norms[:, :, tf.newaxis] +
            token_norms[:, tf.newaxis, :] -
            2 * tf.matmul(tokens, tokens, transpose_b=True))


def dropout_jensen_permutation_loss(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        pi_im = y_pred.get('mixture')
        o_imt = y_pred.get('prediction_logits')

        k = tf.shape(pi_im)[1]
        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, k, 1])

        # (B, k, T)
        log_p_sigma_imt = - pointwise_bce(y_true_tiled, o_imt)

        # (B, k)
        log_p_sigma_im = tf.math.reduce_sum(log_p_sigma_imt, axis=-1)

        # (B,) 
        L_i = - tf.math.reduce_sum(log_p_sigma_im * pi_im, axis=1)
        return tf.math.reduce_mean(L_i)
    return fun

def dropout_permutation_loss(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        u_im = y_pred.get('mixture_logits')
        o_imt = y_pred.get('prediction_logits')

        k = tf.shape(u_im)[1]
        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, k, 1])

        # (B, k, T)
        log_p_sigma_imt = - pointwise_bce(y_true_tiled, o_imt)

        # (B, k)
        log_p_sigma_im = tf.math.reduce_sum(log_p_sigma_imt,
                                            axis=-1)

        log_pi_im = u_im - tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(u_im, axis=-1),
                                                  axis=-1),
                                   (1, k))
        L_i = - tf.math.reduce_logsumexp(log_pi_im + log_p_sigma_im, axis=-1)
        return tf.math.reduce_mean(L_i)
    return fun


def mean_frobnorm_to_mat(mat):
    def fun(y_true, y_pred):
        batchsize = tf.shape(y_pred)[0]
        matrix = tf.tile(tf.expand_dims(tf.dtypes.cast(mat, tf.float32), axis=0), (batchsize, 1, 1))
        return tf.math.reduce_mean(tf.norm(y_pred - matrix, axis=(1, 2)))
    return fun

def mean_frob2korders(soft_orders_in, input_in, orders, M=np.sqrt(3)):
    orders = tf.constant(orders)
    M = tf.dtypes.cast(M, tf.float32)

    def fun(y_true, y_pred):
        K = tf.shape(orders)[0]
        T = tf.shape(orders)[1]
        permutation_matrices = tf.gather(tf.eye(T), orders, axis=0)

        inputs = y_pred[input_in]
        permutation_matrix = y_pred[soft_orders_in]

        batchsize = tf.shape(inputs)[0]
        # Horizontal Separation:
        X_horizontal = tf.tile(tf.expand_dims(inputs[:, 1], axis=1), (1, K))

        # (N, K + 1)
        borders = tf.tile(tf.expand_dims(-M + tf.range(K + 1, dtype=tf.float32) * (2 * M / tf.dtypes.cast(K, dtype=tf.float32)), axis=0), (batchsize, 1))
        bot_borders = borders[:, :-1]
        top_borders = borders[:, 1:]

        # (N, K)
        horizontal_assignment = tf.dtypes.cast(tf.logical_and(bot_borders <= X_horizontal, X_horizontal < top_borders),
                                               dtype=tf.float32)

        horizontal_assignment = tf.tile(tf.reshape(horizontal_assignment, (batchsize, K, 1, 1)),
                                        (1, 1, T, T))
        permutation_matrices = tf.tile(tf.reshape(permutation_matrices, (1, K, T, T)),
                                       (batchsize, 1, 1, 1))
        # (N, T, T)
        ground_truth_orders = tf.math.reduce_sum(horizontal_assignment * permutation_matrices, axis=1)
        return tf.math.reduce_mean(tf.norm(permutation_matrix - ground_truth_orders, axis=(1, 2)))
    return fun

def mean_softorder_matrix(soft_orders_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[soft_orders_in]
        return tf.math.reduce_mean(y_pred, axis=0)
    return fun


def softorder_matrix(soft_orders_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[soft_orders_in]
        return y_pred
    return fun

def mean_norm(pred_in, order):
    def fun(y_true, y_pred):
        # (B, I)
        y_pred = y_pred[pred_in]
        return tf.math.reduce_mean(tf.norm(y_pred, ord=order, axis=-1), axis=0)
    return fun

def l2(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        return (1/2) * tf.math.reduce_sum(tf.math.pow(y_pred, 2))
    return fun
"""
def mean_entropy(mixtures_in):
    def fun(y_true, y_pred):
        # (B, T)
        mixture = y_pred['mixtures_in']
        return tf.math.reduce_mean(tf.math.reduce_sum(mixture * tf.math.log(mixture)/tf.math.log(2),
                                                      axis=1))
    return fun
"""

def mean_tensor(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        return tf.math.reduce_mean(y_pred, axis=0)
    return fun

def mean_entropy(pred_in):
    def fun(y_true, y_pred):
        # (B, T)
        y_pred = y_pred[pred_in]
        entropy_pred = entropy(y_pred)
        return tf.math.reduce_mean(entropy_pred)
    return fun

def entropy(p):
    """
    p: (B, N)
    entropy: (B, )
    """
    bit_normalization = tf.math.log(tf.constant(2, dtype=tf.float32))
    return tf.keras.losses.categorical_crossentropy(p, p, from_logits=False) / bit_normalization

def mean_matrix_entropy(soft_orders_in):
    def fun(y_true, y_pred):
        # (B, T, T)
        soft_orders = y_pred[soft_orders_in]

        # (B, T)
        line_entropy = entropy(soft_orders)
        return tf.math.reduce_mean(line_entropy)
    return fun


def mean_frob2Kseparation(T, K, M=np.sqrt(3)):
    def fun(y_true, y_pred):
        orders = tf.constant(np.array(list(itertools.permutations(range(T))))[:K, :])
        permutation_matrices = tf.gather(tf.eye(T), orders, axis=0)
        inputs = y_pred['input']
        batchsize = tf.shape(inputs)[0]
        permutation_matrix = y_pred['matrix']
        # Horizontal Separation:
        X_horizontal = tf.tile(tf.expand_dims(inputs[:, 1], axis=1), (1, K))

        # (N, K + 1)
        borders = tf.tile(tf.expand_dims(-M + tf.range(K + 1, dtype=tf.float32) * (2 * M / K), axis=0), (batchsize, 1))
        bot_borders = borders[:, :-1]
        top_borders = borders[:, 1:]

        # (N, K)
        horizontal_assignment = tf.dtypes.cast(tf.logical_and(bot_borders <= X_horizontal, X_horizontal < top_borders),
                                               dtype=tf.float32)

        horizontal_assignment = tf.tile(tf.reshape(horizontal_assignment, (batchsize, K, 1, 1)),
                                        (1, 1, T, T))
        permutation_matrices = tf.tile(tf.reshape(permutation_matrices, (1, K, T, T)),
                                       (batchsize, 1, 1, 1))
        # (N, T, T)
        ground_truth_orders = tf.math.reduce_sum(horizontal_assignment * permutation_matrices, axis=1)
        return tf.math.reduce_mean(tf.norm(permutation_matrix - ground_truth_orders, axis=(1, 2)))
    return fun

def closest_permutation_to_K(K, T):
    def fun(bistochastic_matrices):
        """
        bistochastic_matrices (B, T, T)
        """
        batchsize = tf.shape(bistochastic_matrices)[0]
        orders = tf.constant(np.array(list(itertools.permutations(range(T))))[:K, :])
        # (K, T, T)
        permutation_matrices = tf.gather(tf.eye(T), orders, axis=0)
        # (B, K, T, T)
        tiled_permutation_matrices = tf.tile(tf.expand_dims(permutation_matrices, axis=0), (batchsize, 1, 1, 1))
        # (B, K, T, T)
        tiled_bistochastic_matrices = tf.tile(tf.expand_dims(bistochastic_matrices, axis=1), (1, K, 1, 1))
        # (B, K)
        frobnorms = tf.norm(tiled_permutation_matrices - tiled_bistochastic_matrices, axis=(2, 3))
        return tf.math.argmax(-frobnorms, axis=1)
    return fun

def closest_permutation(permutation_matrices):
    def fun(bistochastic_matrices):
        print(type(bistochastic_matrices))
        print(type(permutation_matrices))
        K = tf.shape(permutation_matrices)[0]
        batchsize = tf.shape(bistochastic_matrices)[0]
        tiled_permutation_matrices = tf.tile(tf.expand_dims(permutation_matrices, axis=0), (batchsize, 1, 1, 1))
        # (B, K, T, T)
        tiled_bistochastic_matrices = tf.tile(tf.expand_dims(bistochastic_matrices, axis=1), (1, K, 1, 1))
        # (B, K)
        frobnorms = tf.norm(tiled_permutation_matrices - tiled_bistochastic_matrices, axis=(2, 3))
        return tf.math.argmax(-frobnorms, axis=1)
    return fun


def maonet_frob2toyo(T):
    def fun(y_true, y_pred):
        batchsize = tf.shape(y_true)[0]
        Y_1 = y_true[:, 0]
        canonical_order = tf.norm(y_pred - tf.tile(tf.expand_dims(tf.eye(T), axis=0), (batchsize, 1, 1)),
                                  axis=(1, 2))
        mat = tf.concat([tf.concat([tf.ones((1, 1)), tf.zeros((1, T-1))], axis=1),
                         tf.concat([tf.zeros((T-1, 1)), tf.eye(T - 1)[::-1]], axis=1)],
                         axis=0)
        reversed_order = tf.norm(y_pred - tf.tile(tf.expand_dims(mat, axis=0), (batchsize, 1, 1)),
                                 axis=(1, 2))
        return tf.math.reduce_mean(Y_1 * canonical_order + (1 - Y_1) * reversed_order) 
    return fun

def KL_divergence(y_true, y_pred):
    T = tf.dtypes.cast(tf.shape(y_pred)[1], tf.float32)
    uniform_distribution = tf.ones_like(y_pred) / T
    return tf.keras.losses.KLDivergence()(y_pred, uniform_distribution)

def maonet_frob2toycan(T):
    def fun(y_true, y_pred):
        batchsize = tf.shape(y_true)[0]
        Y_1 = y_true[:, 0]
        canonical_order = tf.norm(y_pred - tf.tile(tf.expand_dims(tf.eye(T), axis=0), (batchsize, 1, 1)),
                                  axis=(1, 2))
        return tf.math.reduce_sum(Y_1 * canonical_order) / tf.math.reduce_sum(Y_1) 
    return fun

def maonet_frob2toyreverse(T):
    def fun(y_true, y_pred):
        batchsize = tf.shape(y_true)[0]
        Y_1 = y_true[:, 0]
        mat = tf.concat([tf.concat([tf.ones((1, 1)), tf.zeros((1, T-1))], axis=1),
                         tf.concat([tf.zeros((T-1, 1)), tf.eye(T - 1)[::-1]], axis=1)],
                         axis=0)
        reversed_order = tf.norm(y_pred - tf.tile(tf.expand_dims(mat, axis=0), (batchsize, 1, 1)),
                                 axis=(1, 2))
        return tf.math.reduce_sum((1 - Y_1) * reversed_order) / tf.math.reduce_sum(1 - Y_1) 
    return fun

def mean_frobnorm_to_id(T):
    def fun(y_true, y_pred):
        batchsize = tf.shape(y_pred)[0]
        mat = tf.tile(tf.expand_dims(tf.eye(T), axis=0), (batchsize, 1, 1))
        return tf.math.reduce_mean(tf.norm(y_pred - mat, axis=(1, 2)))
    return fun

def pointwise_weighted_bce(alpha, beta):
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

def masked_mean_bce(pred_in):
    def fun(y_true, y_pred):
        """
        labels: (..., T)
        logits: (..., T)
        """
        # (..., T)
        y_pred = y_pred[pred_in]
        # (..., T)
        logits = y_pred['logits']
        pred = tf.math.sigmoid(logits)
        # (..., 1)
        padding_mask = tf.expand_dims(y_pred['padding_mask'], axis=-1)
        # tf.print("padding_mask : ", padding_mask[:5, 0])
        # (..., T)
        tw_bce = pointwise_bce(y_true, logits)
        # tf.print("labels : ", y_true[0, 0, : 5])
        # tf.print("predictions : ", pred[0, 0, : 5])

        # (..., T)
        masked_tw_bce = padding_mask * tw_bce
        # (..., )
        bce = tf.math.reduce_sum(masked_tw_bce, axis=-1)
        return tf.math.reduce_mean(bce)
    return fun

def positivelabels_bce(labels, logits):
    """ Compute bce for positive labels only:
    labels: (B, T)
    logits: (B, T)
    """
    positive_rescale = tf.math.reduce_sum(labels, axis=0) + tf.keras.backend.epsilon()
    return tf.math.reduce_mean(tf.math.reduce_sum(labels * pointwise_bce(labels, logits), axis=0)/positive_rescale, axis=0) 
    
def negativelabels_bce(labels, logits):
    """ Compute bce for negative labels only:
    labels: (B, T)
    logits: (B, T)
    """
    negative_rescale = tf.math.reduce_sum(1 - labels, axis=0) + tf.keras.backend.epsilon()
    return tf.math.reduce_mean(tf.math.reduce_sum((1 - labels) * pointwise_bce(labels, logits), axis=0)/negative_rescale, axis=0)

def mtcce(y_true, logits, mask):
    """
    y_true: (B, P, T_tot)
    logits: (B, P, T_tot)
    mask: (T, T_tot)
    """
    # (B, P, T_tot)
    lse = utils.masked_LSE(logits, mask)
    return tf.math.reduce_sum(y_true * (logits - lse), axis=-1)


def semi_supervised_permutation_categorical_loss(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]

        # (B, P)
        mixture_logits = y_pred['mixture_logits']
        # (B, P, C_tot)
        prediction_logits = y_pred['prediction_logits']
        # (T, C_tot)
        task_mask = y_pred['task_mask']
        # (B, T)
        supervision_mask = y_pred['supervision_mask']

        P = tf.shape(mixture_logits)[1]
        C_tot = tf.shape(task_mask)[1]
        T = tf.shape(supervision_mask)[1]
        B = tf.shape(supervision_mask)[0]


        # (B, P)
        log_mixture = mixture_logits - tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(mixture_logits, axis=-1), axis=-1), (1, P))
        print('log_mixture.shape (B, P): ', log_mixture.shape)


        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, P, 1])
        
        # Compute coordinate wise losses regardless of supervision
        # (B, P, C_tot)
        print("prediction_logits.shape : ", prediction_logits.shape)
        lse = utils.masked_LSE(prediction_logits, task_mask)
        print('lse.shape (B, P, C_tot): ', lse.shape)
        # (B, P, C_tot)
        cw_log_pi = y_true_tiled * (prediction_logits - lse)
        print("cw_log_pi.shape (B, P, C_tot) : ", cw_log_pi.shape)
        
        task_mask = tf.tile(tf.reshape(task_mask, (1, 1, T, C_tot)),
                            (B, P, 1, 1))

        # (B, P, T, C_tot)
        masked_cw_log_pi = (tf.tile(tf.expand_dims(cw_log_pi, axis=-2),
                                    (1, 1, T, 1))
                            * task_mask)
        tiled_supervision_mask = tf.tile(tf.reshape(supervision_mask, (B, 1, T, 1)),
                                         (1, P, 1, C_tot))
        # (B, P)
        masked_log_pi = tf.math.reduce_sum(masked_cw_log_pi * tiled_supervision_mask, axis=(-2, -1))

        return - tf.math.reduce_mean(tf.math.reduce_logsumexp(log_mixture + masked_log_pi,
                                                              axis=-1),
                                     axis=0)
    return fun

def tree_permutation_categorical_loss(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        # (B, T!, T)
        log_pi_it = y_pred.get('log_mixture')
        # (B, T!, T_tot)
        logits_pred = y_pred.get('logits_pred')
        task_mask = y_pred.get('task_mask')

        n_perm = tf.shape(logits_pred)[1]
        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])
        log_p_i = mtcce(y_true_tiled, logits_pred, task_mask)
        return tf.math.reduce_mean(-tf.math.reduce_logsumexp(tf.math.reduce_sum(log_pi_it, 
                                                                                axis=-1)
                                                             + log_p_i,
                                                             axis=-1),
                                   axis=-1)
    return fun

def tree_permutation_loss(pred_in):
    def fun(y_true, y_pred):
        # (B, T!, T)
        y_pred = y_pred[pred_in]
        logits_logsumexp = y_pred.get('logits_logsumexp')
        logits_mixture = y_pred.get('logits_mixture')

        # (B, T!, T)
        logits_main = y_pred.get('logits_main')

        n_perm = tf.shape(logits_main)[1]

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])

        # point wise bce of size : (B, T!, T)
        log_p_it = pointwise_bce(y_true_tiled, logits_main)
        log_m_it = logits_mixture - logits_logsumexp
        return tf.math.reduce_mean(-tf.math.reduce_logsumexp(tf.math.reduce_sum(-log_p_it + log_m_it, 
                                                                                axis=-1),
                                                             axis=-1),
                                   axis=-1)
    return fun

def btm_cce(**kwargs):
    categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(**kwargs)

    def fun(y_true, y_pred):
        y_true_multilabel = utils.binary_to_multilabel(y_true)
        return categorical_crossentropy(y_true_multilabel, y_pred)

    return fun

def frequency_balanced_bce(dataset, n_task):
    activated_examples = dataset.reduce(tf.zeros((n_task, ), tf.float32), reducers.activation_count)
    total_examples = dataset.reduce(tf.constant(0, tf.float32), reducers.count_with_batch)
    frequency = activated_examples / total_examples

    alpha = 1 - frequency
    beta = frequency
    return pointwise_weighted_bce(alpha, beta)

def timestep_permutation_loss(timestep):
    def loss(y_true, y_pred):
        mixture = y_pred.get('mixture')
        prediction = y_pred.get('prediction')

        n_perm = tf.shape(mixture)[0]
        y_true_timestep = tf.tile(tf.expand_dims(y_true[:, timestep], axis=1),
                                  multiples=[1, n_perm])

        # shape is (n_perm, )
        pw_bce = tf.math.reduce_mean(pointwise_bce(y_true_timestep, prediction), axis=0)
        return tf.math.reduce_sum(pw_bce * mixture)
    return loss

def timestep_loss(timestep):
    def loss(y_true, y_pred):
        return tf.math.reduce_mean(pointwise_bce(y_true[:, timestep], tf.squeeze(y_pred, axis=-1)))
    return loss


def one_dimensional_gaussian(mean, scale):
    def gaussian(X):
        batch_size = tf.shape(X)[0]
        T = tf.shape(mean)[0]

        X_tiled = tf.tile(tf.expand_dims(X, 1),
                          multiples=[1, T])
        mean_tiled = (tf.tile(tf.expand_dims(mean, 0),
                              multiples=[batch_size, 1]))
        scale_tiled = (tf.tile(tf.expand_dims(scale, 0),
                               multiples=[batch_size, 1])
                       +
                       10)

        quad = (-1/2) * tf.math.pow((X_tiled - mean_tiled)
                                    /
                                    scale_tiled, 2)
        normalisation = 1/(tf.math.sqrt(2 * np.pi)
                           *
                           scale_tiled)
        return normalisation * tf.math.exp(quad)
    return gaussian

def l2_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.maximum(tf.math.reduce_sum(tf.math.square(x), axis=axis), epsilon))

def frob_norm(A):
    return tf.linalg.trace(tf.matmul(tf.transpose(A), A))

def variance(p_i, x_i):
    """returns variance with p_i (N, ) the proba distribution on the x_i (N, )"""
    first_momentum = tf.math.reduce_sum(p_i * x_i)
    second_momentum = tf.math.reduce_sum(p_i * tf.math.pow(x_i, 2))
    return second_momentum - tf.math.pow(first_momentum, 2)

def permutation_variance(y_true, y_pred):
    p_i = y_pred.get('mixture')
    permutation_output = y_pred.get('output')

    n_perm = tf.shape(p_i)[0]

    y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                           multiples=[1, n_perm, 1])

    # pointwise bce of size : (batch_size, n_perm, T)
    pw_bce = pointwise_bce(y_true_tiled, permutation_output)

    # permutation wise bce of size (batch_size, n_perm)
    permw_bce = tf.math.reduce_mean(pw_bce,
                                    axis=-1)
    batch_permw_bce = tf.math.reduce_mean(permw_bce, axis=0)
    return variance(p_i, batch_permw_bce)

def permutation_losses(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        permutation_output = y_pred.get('prediction_logits')
        n_perm = tf.shape(permutation_output)[1]

        # (batch_size, n_perm, T)
        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, n_perm, 1])

        # (batch_size, n_perm, T)
        pw_bce = pointwise_bce(y_true_tiled, permutation_output)
        # (batch_size, n_perm)
        permw_bce = tf.math.reduce_mean(pw_bce,
                                        axis=-1)
        return tf.math.reduce_mean(permw_bce, axis=0)

    return fun


def permutation_mean(y_true, y_pred):
    p_i = y_pred.get('mixture')
    permutation_output = y_pred.get('output')

    n_perm = tf.shape(p_i)[0]

    y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                           multiples=[1, n_perm, 1])

    # pointwise bce of size : (batch_size, n_perm, T)
    pw_bce = pointwise_bce(y_true_tiled, permutation_output)

    # permutation wise bce of size (batch_size, n_perm)
    permw_bce = tf.math.reduce_mean(pw_bce,
                                    axis=-1)
    batch_permw_bce = tf.math.reduce_mean(permw_bce, axis=0)
    return tf.math.reduce_sum(p_i * batch_permw_bce)

def mean_by_batch(y_true, y_pred):
    return tf.math.reduce_mean(y_true, axis=0)

def mean_by_timestep_and_batch(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.reduce_mean(y_pred, axis=0), axis=0)

def permutation_categorical_loss(pred_in):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        # (B, P)
        mixture_logits = y_pred['mixture_logits']
        # (B, P, C_tot)
        prediction_logits = y_pred['prediction_logits']
        # print("prediction_logits.shape (B, P, C_tot) : ", prediction_logits.shape)
        # (T, C_tot)
        task_mask = y_pred['task_mask']

        P = tf.shape(mixture_logits)[1]
        C_tot = tf.shape(task_mask)[1]

        # (B, P)
        log_mixture = mixture_logits - tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(mixture_logits, axis=-1), axis=-1), (1, P))
        # print('log_mixture.shape (B, P): ', log_mixture.shape)

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, P, 1])
        
        # Compute coordinate wise losses regardless of supervision
        # (B, P, C_tot)
        lse = utils.masked_LSE(prediction_logits, task_mask)
        # print('lse.shape (B, P, C_tot): ', lse.shape)
        # (B, P, C_tot)
        coordwise_log_pi = y_true_tiled * (prediction_logits - lse)
        # (B, P)
        permwise_log_pi = tf.math.reduce_sum(coordwise_log_pi, axis=-1)
        
        return - tf.math.reduce_mean(tf.math.reduce_logsumexp(log_mixture + permwise_log_pi,
                                                              axis=-1),
                                     axis=0)
    return fun

def weighted_permutation_categorical_loss(pred_in, task_weights):
    task_weights = tf.constant(task_weights, dtype=tf.float32)

    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        # (B, P)
        mixture_logits = y_pred['mixture_logits']
        # (B, P, C_tot)
        prediction_logits = y_pred['prediction_logits']
        # print("prediction_logits.shape (B, P, C_tot) : ", prediction_logits.shape)
        # (T, C_tot)
        task_mask = y_pred['task_mask']

        B = tf.shape(mixture_logits)[0]
        P = tf.shape(mixture_logits)[1]
        C_tot = tf.shape(task_mask)[1]
        T = tf.shape(task_mask)[0]

        # (B, P)
        log_mixture = mixture_logits - tf.tile(tf.expand_dims(tf.math.reduce_logsumexp(mixture_logits, axis=-1), axis=-1), (1, P))
        # print('log_mixture.shape (B, P): ', log_mixture.shape)

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, P, 1])
        
        # Compute coordinate wise losses regardless of supervision
        # (B, P, C_tot)
        lse = utils.masked_LSE(prediction_logits, task_mask)
        # print('lse.shape (B, P, C_tot): ', lse.shape)
        # (B, P, C_tot)
        coordwise_log_pi = y_true_tiled * (prediction_logits - lse)

        # (B, P, T, C_tot)
        tiled_coordwise_log_pi = tf.tile(tf.reshape(coordwise_log_pi, (B, P, 1, C_tot)),
                                         (1, 1, T, 1))
        # (B, P, T, C_tot)
        tiled_mask = tf.tile(tf.reshape(task_mask, (1, 1, T, C_tot)), (B, P, 1, 1))

        # (B, P, T)
        taskwise_log_pi = tf.math.reduce_sum(tiled_mask * tiled_coordwise_log_pi, axis=-1)
        tiled_weights = tf.tile(tf.reshape(task_weights, (1, 1, T)), (B, P, 1))
        permwise_log_pi = tf.math.reduce_sum(tiled_weights * taskwise_log_pi, axis=-1)
        
        return - tf.math.reduce_mean(tf.math.reduce_logsumexp(log_mixture + permwise_log_pi,
                                                              axis=-1),
                                     axis=0)
    return fun

def weighted_categorical_permutation_dice(pred_in, task_weights, epsilon=1e-7):
    epsilon = tf.constant(epsilon, dtype=tf.float32)
    task_weights = tf.constant(task_weights, dtype=tf.float32)
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        # (B, P)
        mixture = tf.nn.softmax(y_pred['mixture_logits'])
        # (B, P, C_tot)
        prediction_logits = y_pred['prediction_logits']
        task_mask = y_pred['task_mask']
        # print("prediction_logits.shape (B, P, C_tot) : ", prediction_logits.shape)
        # (T, C_tot)

        B = tf.shape(mixture)[0]
        P = tf.shape(mixture)[1]
        T = tf.shape(prediction_logits)[-1] // 2 

        # print('log_mixture.shape (B, P): ', log_mixture.shape)

        y_true_tiled = tf.tile(tf.expand_dims(y_true, 1),
                               multiples=[1, P, 1])
        
        # (B, P, C_tot)
        lse = utils.masked_LSE(prediction_logits, task_mask)
        prediction = tf.math.exp(prediction_logits - lse)
        # Select only the first coefficient i.e proba that AU is present.
        # (B, P, T)
        hat_p_i = tf.reshape(prediction, (B, P, T, 2))[:, :, :, 0]

        # (B, P, T)
        p_i = tf.reshape(y_true_tiled, (B, P, T, 2))[:, :, :, 0]

        # (B, P, T)
        coordwise_dice = 1 - (2 * p_i * hat_p_i + epsilon) / (tf.math.square(p_i)
                                                              + tf.math.square(hat_p_i)
                                                              + epsilon)
        # (B, P, T)
        tiled_weights = tf.tile(tf.reshape(task_weights, (1, 1, T)), (B, P, 1))

        # (B, P)
        permwise_dice = tf.math.reduce_sum(tiled_weights * coordwise_dice, axis=-1)
        elementwise_dice = tf.math.reduce_sum(permwise_dice * mixture, axis=-1)
        return tf.math.reduce_mean(elementwise_dice)
    return fun

def weighted_binary_permutation_loss(pred_in, task_weights):
    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]

        # (B, k)
        pi_im = y_pred.get('mixture')
        # (B, k, T)
        o_imt = y_pred.get('prediction_logits')

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
        return tf.math.reduce_mean(L_i)
    return fun

def weighted_binary_permutation_dice(pred_in, task_weights, epsilon=1.0):
    epsilon = tf.constant(epsilon, dtype=tf.float32)

    def fun(y_true, y_pred):
        y_pred = y_pred[pred_in]
        # (B, P)
        mixture = y_pred['mixture']
        # (B, P, T)
        prediction_logits = y_pred['prediction_logits']
        prediction = tf.nn.sigmoid(prediction_logits)

        B = tf.shape(mixture)[0]
        P = tf.shape(mixture)[1]
        T = tf.shape(prediction_logits)[-1]

        # (B, P, T)
        p_i = tf.tile(tf.expand_dims(y_true, 1),
                      multiples=[1, P, 1])

        # (B, P, T)
        hat_p_i = prediction

        # (B, P, T)
        coordwise_dice = 1 - (2 * p_i * hat_p_i + epsilon) / (tf.math.square(p_i)
                                                              + tf.math.square(hat_p_i)
                                                              + epsilon)
        # (B, P, T)
        tiled_weights = tf.tile(tf.reshape(task_weights, (1, 1, T)), (B, P, 1))

        # (B, P)
        permwise_dice = tf.math.reduce_sum(tiled_weights * coordwise_dice, axis=-1)
        elementwise_dice = tf.math.reduce_sum(permwise_dice * mixture, axis=-1)
        return tf.math.reduce_mean(elementwise_dice)
    return fun

if __name__ == '__main__':
    pass




