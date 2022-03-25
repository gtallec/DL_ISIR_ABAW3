import tensorflow as tf
from mappers import label_projection


@tf.function
def count(x, el):
    return x + 1

def activation_count(x, el):
    sum_el = tf.math.reduce_sum(el, axis=0)
    return x + sum_el 

def count_with_batch(x, el):
    return x + tf.dtypes.cast(tf.shape(el)[0],
                              tf.float32)

def get_label(image, label):
    return label

def pairwise_joint_distribution(i, j):
    def fun(dataset):
        joint_dataset = (dataset
                         .map(label_projection([i, j]), tf.data.experimental.AUTOTUNE)
                         .map(unfold_joint_map, tf.data.experimental.AUTOTUNE))
        return distribution(joint_dataset, 2)
    return fun

def pairwise_marginal_product(i, j):
    def fun(dataset):
        n_el = dataset.reduce(tf.constant(0, tf.float32), count_with_batch)
        p_i = ((dataset
                .map(label_projection([i]), tf.data.experimental.AUTOTUNE)
                .reduce(tf.constant(0, tf.float32), activation_count))/n_el)[0]
        tf.print('p_i : ', p_i)
        p_j = ((dataset
                .map(label_projection([j]), tf.data.experimental.AUTOTUNE)
                .reduce(tf.constant(0, tf.float32), activation_count))/n_el)[0]

        tf.print('p_j : ', p_j)
        tf.print([(1 - p_i) * (1 - p_j), p_i * (1 - p_j), (1 - p_i) * p_j, p_j * p_i])
        return [(1 - p_i) * (1 - p_j), p_i * (1 - p_j), (1 - p_i) * p_j, p_j * p_i]
    return fun

def paiwise_mutual_information(i, j):
    def fun(dataset):
        dataset_pairwise_marginal_product = pairwise_marginal_product(i, j)(dataset)
        dataset_pairwise_joint_distribution = pairwise_joint_distribution(i, j)(dataset)
        return tf.keras.losses.KLDivergence()(dataset_pairwise_joint_distribution,
                                              dataset_pairwise_marginal_product)
    return fun


def unfold_joint_map(x):

    batchsize, x_shape = tf.shape(x)[0], tf.shape(x)[1]
    to_exp = 2 * tf.ones((x_shape,), dtype=tf.int32)
    exp = tf.range(start=0, limit=x_shape, delta=1)
    base_2 = tf.dtypes.cast(tf.pow(to_exp, exp),
                            dtype=tf.float32)
    indices = tf.matmul(x, tf.expand_dims(base_2, axis=1))
    a = tf.tile(tf.expand_dims(tf.range(tf.pow(2, x_shape)),
                               axis=0),
                [batchsize, 1])
    res = tf.dtypes.cast(indices - tf.dtypes.cast(a, tf.float32) == 0, tf.float32)
    return res
    
def pointwise_sum(initial_state, new_element):
    return initial_state + tf.math.reduce_sum(new_element,
                                              axis=0)

def distribution(dataset, n_dim):
    n_el = dataset.reduce(tf.constant(0, tf.float32), count_with_batch)
    return dataset.reduce(tf.zeros((tf.pow(2, n_dim),), dtype=tf.float32), pointwise_sum)/n_el

def binary_marginal_product(marginals):
    """ Computes the product of N binary marginals.
    marginals contain N marginals of binary variables in a matrix (N,) where each element is p_i = P(X_i = 1)"""
    N = tf.shape(marginals)[0]
    binary_combinations = binary_combination(N)[:, ::-1]






    

    


def binary_combination(n):
    to_decomp = tf.range(start=0, limit=(2 ** n), delta=1)
    base = (2 ** (n - 1)) * tf.ones((2 ** n, ), dtype=tf.int32)
    decomp = tf.ones((2 ** n, 0), dtype=tf.int32)
    for i in range(n):
        (q, r) = (tf.math.floordiv(to_decomp, base),
                  tf.math.floormod(to_decomp, base))
        decomp = tf.concat([decomp, tf.expand_dims(q,
                                                   axis=1)], axis=1)
        to_decomp = r
        base = tf.math.floordiv(base, 2)

    return decomp

if __name__ == '__main__':
    tf.print(binary_decomposition(6))
    a = tf.reshape(tf.range(start=0, limit=8, delta=1), (4, 2))
    b = tf.expand_dims(tf.range(start=0, limit=4, delta=1),
                       1)
    indices = tf.expand_dims(tf.constant([1, 0, 1, 0], tf.int32), 1)

    b_indices = tf.concat([b, indices], axis=1)
    tf.print(b_indices)
    tf.print(a)
    tf.print(tf.gather_nd(a, b_indices))
    a = tf.range(start=0, limit=31, delta=1)



