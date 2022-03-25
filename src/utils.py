import numpy as np
import tensorflow as tf
import json
import itertools
import matplotlib.pyplot as plt


def dict_from_json(json_file):
    with open(json_file) as f:
        json_dict = json.load(f)
    return json_dict

def expand_to_blocks(permutation_matrix, len_blocks):
    N = permutation_matrix.shape[0]
    permutation_rows = []
    for i in range(N):
        block_rows = []
        # Search for which block is treated the i-th 
        p = 0
        while (p < N) and (permutation_matrix[i][p] == 0):
            p += 1
        B_p = len_blocks[p]
        for i in range(N):
            B_i = len_blocks[i]
            if i != p:
                block_rows.append(np.zeros((B_p, B_i)))
            else:
                block_rows.append(np.identity(B_p))
        permutation_rows.append(np.concatenate(block_rows, axis=1))
    permutation = np.concatenate(permutation_rows, axis=0)
    return permutation

def relative_block_coords(block):
    return recursive_block(block, 0)[1]

def recursive_block(block, n_el):
    L = []
    print(block)
    for i in range(len(block)):
        if isinstance(block[i], list):
            nblock_el, new_block = recursive_block(block[i], n_el)
            L.append(new_block)
            n_el = nblock_el
        else:
            L.append(n_el)
            n_el += 1
    print(n_el)
    return n_el, L

def sum_block(block, L):
    for i in range(len(block)):
        if isinstance(block[i], list):
            L = sum_block(block[i], L)
        else:
            L.append(block[i])
    return L

@tf.function
def binary_to_multilabel(vector):
    print(tf.shape(vector))
    shape = tf.shape(vector)
    batchsize = shape[0]
    N = shape[1]
    print('batchsize : ', batchsize)
    print('N : ', N)
    
    exp = tf.tile(tf.expand_dims(tf.math.pow(2, tf.range(0, N)), axis=0),
                  multiples=[batchsize, 1])
    exp = tf.dtypes.cast(exp, dtype=tf.float32)
    label_index = tf.math.reduce_sum(exp * vector, axis=1)
    label_index = tf.tile(tf.expand_dims(label_index, axis=1), multiples=[1, 2**N])
    label_matching = tf.expand_dims(tf.range(0, tf.pow(2, N)), axis=0)
    label_matching = tf.tile(label_matching, multiples=[batchsize, 1])
    label_matching = tf.dtypes.cast(label_matching, tf.float32)
    labels = tf.dtypes.cast(label_index - label_matching == 0, dtype=tf.float32)

    return labels

def multilabel_to_binary_matrix(N):
    matrix_rows = []
    cur = tf.range(0, tf.pow(2, N))
    for i in range(N-1, -1, -1):
        quotient = tf.math.floordiv(cur, tf.pow(2, i))
        cur = tf.math.floormod(cur, tf.pow(2, i))
        matrix_rows.append(tf.expand_dims(tf.dtypes.cast(quotient == 1, tf.float32), axis=0))
    
    return tf.concat(matrix_rows[::-1], axis=0)

def sample_onehot_from_categorical(categoricals):
    # categorical of size (batchsize, N, n_class)
    shape = tf.shape(categoricals)
    sample_size = shape[:-1]
    n_class = shape[-1:]
    lower_gather = tf.range(0, tf.squeeze(n_class))
    upper_gather = tf.range(1, tf.squeeze(n_class) + 1)

    # (batchsize, N, n_class)
    bins = tf.concat([tf.zeros(tf.concat([sample_size, [1]], axis=0)), tf.math.cumsum(categoricals, axis=-1)], axis=-1)
    samples = tf.random.uniform(shape=tf.concat([sample_size, [1]], axis=0),
                                minval=0,
                                maxval=1)

    tiled_samples = tf.tile(samples, multiples=tf.concat([tf.ones_like(sample_size), n_class], axis=0))
    upper_cum = tf.dtypes.cast(tiled_samples - tf.gather(bins, upper_gather, axis=-1) < 0, dtype=tf.float32)
    lower_cum = tf.dtypes.cast(tiled_samples - tf.gather(bins, lower_gather, axis=-1) >= 0, dtype=tf.float32)
    return upper_cum * lower_cum

def sample_from_categorical(categoricals):
    one_hots = sample_onehot_from_categorical(categoricals)
    shape = tf.shape(categoricals)
    sample_size = shape[:-1]
    n_class = shape[-1:]
    classes = tf.reshape(tf.range(0, tf.squeeze(tf.dtypes.cast(n_class, dtype=tf.float32)), dtype=tf.float32), tf.concat([tf.ones_like(sample_size), n_class], axis=0))
    classes = tf.tile(classes, tf.concat([sample_size, tf.ones_like(n_class)], axis=0))

    return tf.dtypes.cast(tf.math.reduce_sum(classes * one_hots, axis=-1), dtype=tf.int32)

def sample_without_replacement(logits, K):
    """
    Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    """
    z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    _, indices = tf.nn.top_k(logits + z, K)
    return indices

def masked_LSE(logits, mask):
    """ 
    logits : (..., T_tot),
    mask : (T, T_tot)
    """
    T, T_tot = tf.shape(mask)[0], tf.shape(mask)[1]
    batchsize = tf.shape(logits)[:-1]
    # (..., T, T_tot)
    mask = tf.tile(tf.reshape(mask, tf.concat([tf.ones_like(batchsize), tf.shape(mask)], axis=0)),
                   tf.concat([batchsize, tf.ones_like(tf.shape(mask))], axis=0))
    # (..., T, T_tot)
    logits = tf.tile(tf.expand_dims(logits, axis=-2), tf.concat([tf.ones_like(batchsize), [T, 1]], axis=0))
    
    # (..., T, T_tot)
    logits_masked = logits * mask
    logits_max = tf.tile(tf.expand_dims(tf.math.reduce_max(logits_masked, axis=-1), axis=-1),
                         tf.concat([tf.ones_like(batchsize), [1, T_tot]], axis=0))

    exp_logits_masked = tf.math.exp(logits - logits_max) * mask

    # (..., T, T_tot)
    logsumexp = logits_max + tf.tile(tf.expand_dims(tf.math.log(tf.math.reduce_sum(exp_logits_masked, axis=-1)),
                                                    axis=-1),
                                     tf.concat([tf.ones_like(batchsize), [1, T_tot]], axis=0))

    return tf.math.reduce_sum(logsumexp * mask, axis=-2)

def make_balanced_video_repartition(activations,
                                    counts,
                                    fold_repartition,
                                    N):
    """
    activations (V, T): activations[v][t] is the number of class t examples in video v.
    counts (V, ): counts[v] is the number of frames in video v.
    k: Number of folds to make.
    N: Beam Search explor/exploit arguments.
    """
    V = counts.shape[0]
    k = len(fold_repartition)
    """
        fold_repartition = k * [V // k]
    else:
        fold_repartition = (k - 1) * [V // (k-1)] + [V % (k-1)]
    """
    # (T, )
    p = np.sum(activations, axis=0) / np.sum(counts)
    print("p : ", p)

    # (1, V)
    fold_assignment_i = np.zeros((1, V))
    # (1, V)
    for i in range(k - 1):
        mask_assignment_i = fold_assignment_i >= k - 1 - i

        losses = []
        fold_assignment_ip1 = []
        for j in range(fold_assignment_i.shape[0]):
            # (V, )
            fold_assignment_ij = fold_assignment_i[j, :]
            mask_assignment_ij = mask_assignment_i[j, :] 
            unassigned_videos_ij = np.arange(V)[~mask_assignment_ij]
            mask_assignment_ij = mask_assignment_ij.astype(float)
            
            # (f_i parmi V - F_{<i}, f_i)
            combination = np.array(list(itertools.combinations(unassigned_videos_ij, fold_repartition[i])))
            # (f_i parmi V - F_{<i}, V)
            combination = np.sum(np.identity(V)[combination, :], axis=1)
            # (f_i parmi V - F_{<i}, V)
            mask_assignment_ip1j = mask_assignment_ij[np.newaxis, :] + combination
            # (f_i parmi V - F_{<i}, V)
            fold_assignment_ip1j = fold_assignment_ij[np.newaxis, :] + (k - 1 - i) * combination

            
            # (f_i parmi V - F_{<i}, T)
            combination_activations = np.sum(combination[:, :, np.newaxis] * activations[np.newaxis, :], axis=1)
            # (f_i parmi V - F_{<i}, 1)
            combination_counts = np.sum(combination * counts[np.newaxis, :], axis=-1)[:, np.newaxis]
 
            # (f_i parmi V - F_{<i}, T)
            C_combination_activations = np.sum((1 - mask_assignment_ip1j)[:, :, np.newaxis] * activations[np.newaxis, :], axis=1)
            # (f_i parmi V - F_{<i}, 1)
            C_combination_counts = np.sum((1 - mask_assignment_ip1j) * counts[np.newaxis, :], axis=-1)[:, np.newaxis]

            # (f_i parmi V - F_{<i}, T)
            p_ij = combination_activations / combination_counts
            print('p_{}{}'.format(i, j), p_ij[0])
            # (f_i parmi V - F_{<i}, T)
            C_p_ij = C_combination_activations / C_combination_counts

            loss = np.sum(np.power(p - p_ij, 2) + np.power(p - C_p_ij, 2), axis=1)

            losses.append(loss)
            fold_assignment_ip1.append(fold_assignment_ip1j)
       
        # (N x (f_i parmi V - F_{<i}), )
        losses = np.concatenate(losses, axis=0)
        sort_indices = np.argsort(losses)[:N]
        # (N x (f_i parmi V - F_{<i}), V)
        fold_assignment_ip1 = np.concatenate(fold_assignment_ip1, axis=0)
        fold_assignment_i = fold_assignment_ip1[sort_indices]
    return fold_assignment_i[0]
        

if __name__ == '__main__':
    pass






    


    



