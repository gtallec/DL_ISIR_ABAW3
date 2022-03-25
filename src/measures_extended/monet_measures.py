import tensorflow as tf

def frobnorm_to_mat(soft_order, mat):
    def fun(y_true, y_pred):
        # (T, T)
        y_pred = y_pred[soft_order]
        return tf.norm(y_pred - mat)
    return fun

