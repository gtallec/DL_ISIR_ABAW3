import tensorflow as tf


def task_conditionning(mat):
    n_task = tf.shape(mat)[-1]
    mat_tiled = tf.tile(tf.expand_dims(mat, -1), multiples=tf.concat([tf.ones_like(tf.shape(mat)), [n_task]], axis=0))
    id_mat = tf.reshape(tf.eye(n_task), tf.concat([tf.ones_like(tf.shape(mat)[:-1]), [n_task, n_task]], axis=0))
    id_mat = tf.tile(id_mat, multiples=tf.concat([tf.shape(mat)[:-1], [1, 1]], axis=0))
    t_rank = tf.rank(mat)
    transpose_perm = tf.concat([tf.range(t_rank - 1), [t_rank, t_rank - 1]], axis=0)
    mat_tiled = tf.transpose(mat_tiled * (1 - id_mat), transpose_perm)
    mat_tiled = tf.reshape(mat_tiled, tf.concat([tf.shape(mat)[:-1], [n_task * n_task]], axis=0))
    mat_tiled = tf.reshape(tf.boolean_mask(mat_tiled, mat_tiled != 0), tf.concat([tf.shape(mat)[:-1], [n_task - 1, n_task]], axis=0))
    mat_tiled = tf.transpose(mat_tiled, transpose_perm)
    # shape = tf.shape(mat_tiled)
    # mat_tiled = tf.reshape(mat_tiled, tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0))

    return mat_tiled
