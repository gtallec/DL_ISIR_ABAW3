import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

from datasets.toys.config import SAVE_NORM_PATH
from datasets.toys.toy_utils import bernouilli

def parametrized_dependency(N, p, p_c, mode='train', batchsize=32, seed=None, meta=False, **kwargs):
    """
    Input are random noise
    Task 1 is bernouilli with parameter p
    Task 2 is Task 1 with probability 1 - p_c and (1 - Task 1) with probability p_c.
    I(Task 1, Task 2) = H(p_c + p - 2*(p_c)*p) - (1 - p) * H(p_c) - p * H(1 - p_c)
    """
    np.random.seed(seed)
    X = np.random.random(size=(N, 299, 299, 3))
    task1 = tf.dtypes.cast(bernouilli(p=p*np.ones((N, ))), tf.float32)
    label_corruption = tf.dtypes.cast(bernouilli(p_c * np.ones((N, ))), tf.float32)
    task2 = ((2 * (1 - label_corruption) - 1) * (2 * task1 - 1) + 1)/2
    labels = tf.stack([task1, task2], axis=1)

    save_file = os.path.join(SAVE_NORM_PATH, 'parametrized_dependency', 'normalisation.npz')
    if mode == 'train':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        np.savez(save_file,
                 mean=mean,
                 std=std)
    elif mode in ['test', 'eval']:
        normalisation = np.load(save_file)
        mean = normalisation['mean']
        std = normalisation['std']

    eps = 1e-7
    features = ((X - mean) / (std + eps))

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    if meta:
        dataset = tf.data.Dataset.from_tensor_slices(labels)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    return (dataset
            .shuffle(N)
            .batch(batchsize))


SUPPORTED_ATTENTION_TOYS = {"dependency": parametrized_dependency}

if __name__ == '__main__':
    pass
