import tensorflow as tf
import os

def count(dataset):
    def count_with_batch(x, el):
        return x + tf.dtypes.cast(tf.shape(el)[0],
                                  tf.int64)
    return dataset.reduce(tf.constant(0, tf.int64), count_with_batch)

def count_batches(dataset):
    def fun(x, el):
        return x + 1
    return dataset.reduce(tf.constant(0, tf.int64), fun)

def occurences(dataset):
    label_shape = tf.shape(next(iter(dataset)))[1]
    def count_activations_with_batch(x, el):
        return x + tf.math.reduce_sum(el, axis=0)

    return dataset.reduce(tf.constant(tf.zeros((label_shape, )), dtype=tf.float32),
                          count_activations_with_batch)

def frequencies(dataset):
    return occurences(dataset)/tf.dtypes.cast(count(dataset), tf.float32)

def partial_labels_count(dataset):

    T = tf.shape(next(iter(dataset)))[1]

    def partial_labels_count_with_batch(x, el):
        """ el of size (B, T) """
        mask = 1 - tf.dtypes.cast(el == -1, dtype=tf.float32)
        return x + tf.math.reduce_sum(tf.ones_like(el, dtype=tf.float32) * mask, axis=0)

    return dataset.reduce(tf.constant(tf.zeros((T, )), dtype=tf.float32),
                          partial_labels_count_with_batch)

def partial_labels_occurences(dataset):

    T = tf.shape(next(iter(dataset)))[1]

    def partial_labels_count_activations_with_batch(x, el):
        mask = 1 - tf.dtypes.cast(el == -1, dtype=tf.float32)
        return x + tf.math.reduce_sum(tf.dtypes.cast(el, tf.float32) * mask, axis=0)

    return dataset.reduce(tf.constant(tf.zeros((T, )), dtype=tf.float32),
                          partial_labels_count_activations_with_batch)

def partial_labels_frequencies(dataset):
    return (partial_labels_occurences(dataset)) / (partial_labels_count(dataset) + 1e-7)

SUPPORTED_STATISTICS = {"count": count,
                        "plcount": partial_labels_count,
                        "occurences": occurences,
                        "ploccurences": partial_labels_occurences,
                        "frequencies": frequencies,
                        "plfrequencies": partial_labels_frequencies,
                        "steps_by_epoch": count_batches}

def get_dataset_statistic(dataset_statistic):
    return SUPPORTED_STATISTICS[dataset_statistic]

if __name__ == '__main__':
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant([[[1, 2, 3]],
                                                               [[4, 5, 6]]],
                                                              dtype=tf.float32)))
    tf.print(count_occurences(dataset))
