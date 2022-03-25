import tensorflow as tf

def batch_matmul(mat, batch):
    expanded_batch = tf.expand_dims(batch, axis=-1)
    return tf.squeeze(tf.matmul(mat, expanded_batch), axis=-1)

def multi_gather(params, indices):
    batchsize = tf.shape(params)[0]
    select = tf.shape(indices)[1]
    # (B, S)
    p_axis = tf.tile(tf.expand_dims(tf.range(0, batchsize), axis=1), (1, select))
    indices = tf.concat([tf.expand_dims(p_axis, -1), tf.expand_dims(indices, axis=-1)], axis=-1)
    return tf.gather_nd(params, indices)


