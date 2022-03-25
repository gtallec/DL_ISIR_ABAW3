import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

from datasets.toys.config import SAVE_NORM_PATH
from datasets.toys.toy_utils import bernouilli

def fractal_separation(N, n_task, mode='train', batchsize=32):
    X = 2 * np.random.random(size=(N, 2)) - 1
    labels = np.zeros((N, n_task))
    for i in range(1, n_task + 1):
        X_tiled = np.tile(X[:, np.newaxis, :], 
                          reps=(1, 2 ** (i), 1)) 
        b = 2 * np.arange(2 ** (i) + 1) / (2 ** (i)) - 1
        b = np.tile(b[np.newaxis, :], reps=(N, 1))
        left, right = b[:, :-1], b[:, 1:] 
        rel_coord = X_tiled[:, :, 0] # - A * np.sin(2 * np.pi * X_tiled[:, :, 1]/T)
        demi_plane_bel = np.logical_and(left < rel_coord, rel_coord <= right)
        labels[:, i-1] = np.any(demi_plane_bel[:, ::2], axis=1)


    save_file = os.path.join(SAVE_NORM_PATH, 'fractal_linsep', 'normalisation.npz')
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
    features = ((X - mean)
                /
                (std + eps))

    """
    X_coord, Y_coord = X[:, 0], X[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=n_task, sharey=True, figsize=(5 * (n_task - 2), 3))
    for i in range(n_task):
        axes[i].set_title('Task {}'.format(i))
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        axes[i].scatter(X_coord[pos], Y_coord[pos], s=0.5, color='lime')
        axes[i].scatter(X_coord[neg], Y_coord[neg], s=0.5, color='crimson')
    plt.show()
    """

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batchsize))


def fractal_separation_for_visu(N, n_task):
    X = 2 * np.random.random(size=(N, 2)) - 1
    labels = np.zeros((N, n_task))
    for i in range(1, n_task + 1):
        X_tiled = np.tile(X[:, np.newaxis, :], 
                          reps=(1, 2 ** (i), 1)) 
        b = 2 * np.arange(2 ** (i) + 1) / (2 ** (i)) - 1
        b = np.tile(b[np.newaxis, :], reps=(N, 1))
        left, right = b[:, :-1], b[:, 1:] 
        rel_coord = X_tiled[:, :, 0] # - A * np.sin(2 * np.pi * X_tiled[:, :, 1]/T)
        demi_plane_bel = np.logical_and(left < rel_coord, rel_coord <= right)
        labels[:, i-1] = np.any(demi_plane_bel[:, ::2], axis=1)

    """
    X_coord, Y_coord = X[:, 0], X[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=n_task, sharey=True, figsize=(5 * (n_task - 2), 3))
    for i in range(n_task):
        axes[i].set_title('Task {}'.format(i))
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        axes[i].scatter(X_coord[pos], Y_coord[pos], s=0.5, color='lime')
        axes[i].scatter(X_coord[neg], Y_coord[neg], s=0.5, color='crimson')
    plt.show()
    """

    return X, labels

def D_separation(D, N, k_min, k_max, mode='train', batchsize=32):
    X = np.random.random(size=(N, ))
    width = 1 / (2 ** (k_max + 1))
    bins = (2 * np.arange(2 ** k_max) + 1) * width
    bins_tiled = np.tile(bins[np.newaxis, :], (N, 1))
    X_tiled = np.tile(X[:, np.newaxis], (1, 2 ** k_max))
    bins_assignment = np.abs(X_tiled - bins_tiled) < width
    print(bins_assignment)

    labels = np.zeros((N, k_max - k_min))
    labels[:, k_max - k_min - 1] = np.any(bins_assignment[:, ::2], axis=1)

    for i in range(k_max - k_min - 2, -1, -1):
        bins_assignment = np.reshape(bins_assignment, (N, 2 ** (i + k_min + 1), 2))
        bins_assignment = np.any(bins_assignment, axis=2)
        labels[:, i] = np.any(bins_assignment[:, ::2], axis=1)

    """
    X_coord, Y_coord = X, X
    fig, axes = plt.subplots(nrows=1, ncols=k_max - k_min - 1, figsize=(5 * (k_max - k_min), 3))
    for i in range(k_max - k_min - 1):
        axes[i].set_title('Task {}'.format(i))
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        axes[i].scatter(X_coord[pos], Y_coord[pos], s=0.5, color='lime')
        axes[i].scatter(X_coord[neg], Y_coord[neg], s=0.5, color='crimson')
    plt.show()
    """

    X = np.tile(X[:, np.newaxis], (1, D)) * np.ones((N, D))

    save_file = os.path.join(SAVE_NORM_PATH, 'D_separation', 'normalisation.npz')
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
    features = ((X - mean)
                /
                (std + eps))
    print("mean : ", np.mean(features, axis=0))
    print('std : ', np.std(features, axis=0))

    np.random.seed(None)

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batchsize))



def fractal_separationv2(N, k_min, k_max, M=1, mode='train', batchsize=32, seed=None, **kwargs):
    np.random.seed(seed)
    X = M * (2 * np.random.random(size=(N, 2)) - 1)
    labels = np.zeros((N, k_max - k_min))
    for i in range(k_min, k_max + 1):
        X_tiled = np.tile(X[:, np.newaxis, :], 
                          reps=(1, 2 ** (i), 1)) 
        b = M * (2 * np.arange(2 ** (i) + 1) / (2 ** (i)) - 1)
        b = np.tile(b[np.newaxis, :], reps=(N, 1))
        left, right = b[:, :-1], b[:, 1:] 
        rel_coord = X_tiled[:, :, 0] # - A * np.sin(2 * np.pi * X_tiled[:, :, 1]/T)
        demi_plane_bel = np.logical_and(left < rel_coord, rel_coord <= right)
        labels[:, i - 1 - k_min] = np.any(demi_plane_bel[:, ::2], axis=1)


    save_file = os.path.join(SAVE_NORM_PATH, 'fractal_linsep', 'normalisation.npz')
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
    features = ((X - mean)
                /
                (std + eps))

    np.random.seed(None)

    """
    X_coord, Y_coord = features[:, 0], features[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=k_max - k_min, sharey=True, figsize=(5 * (k_max - k_min - 2), 3))
    for i in range(k_max - k_min):
        axes[i].set_title('Task {}'.format(i))
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        axes[i].scatter(X_coord[pos], Y_coord[pos], s=0.5, color='lime')
        axes[i].scatter(X_coord[neg], Y_coord[neg], s=0.5, color='crimson')
    plt.show()
    """

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batchsize))


def fractal_separation_maonet(N, k_min, k_max, M=1, p=0.0, visu=False, mode='train', batchsize=32, seed=None):
    np.random.seed(seed)
    X = M * (2 * np.random.random(size=(N, 2)) - 1)
    labels = np.zeros((N, k_max - k_min))
    for i in range(k_min, k_max + 1):
        X_tiled = np.tile(X[:, np.newaxis, :], 
                          reps=(1, 2 ** (i), 1)) 
        b = M * (2 * np.arange(2 ** (i) + 1) / (2 ** (i)) - 1)
        b = np.tile(b[np.newaxis, :], reps=(N, 1))
        left, right = b[:, :-1], b[:, 1:] 
        rel_coord = X_tiled[:, :, 0] # - A * np.sin(2 * np.pi * X_tiled[:, :, 1]/T)
        demi_plane_bel = np.logical_and(left < rel_coord, rel_coord <= right)
        labels[:, i - 1 - k_min] = np.any(demi_plane_bel[:, ::2], axis=1)

    canonical_order = np.tile(np.arange(k_max - k_min)[np.newaxis, :],
                              (N, 1))
    reversed_order = np.tile(np.arange(k_max - k_min)[::-1][np.newaxis, :],
                             (N, 1))
    bernouilli_samples = 2 * bernouilli(p * np.ones((N, ))) - 1
    Y_switch = (X[:, 1] > 0).astype(np.float)
    Y_switch = ((bernouilli_samples * (2 * Y_switch - 1)) + 1)/2
    Y_aux = np.tile(Y_switch[:, np.newaxis], (1, k_max - k_min))

    order = Y_aux * canonical_order + (1 - Y_aux) * reversed_order
    order_for_gather = np.concatenate([np.tile(np.arange(N)[:, np.newaxis], (1, k_max - k_min))[:, :, np.newaxis],
                                       order[:, :, np.newaxis]], axis=-1)
    order_for_gather = tf.constant(order_for_gather, dtype=tf.int32)


    save_file = os.path.join(SAVE_NORM_PATH, 'fractal_linsep_maonet', 'normalisation.npz')
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
    features = ((X - mean)
                /
                (std + eps))

    np.random.seed(None)


    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)
    Y_switch = tf.constant(Y_switch, dtype=tf.float32)
    labels = tf.concat([tf.expand_dims(Y_switch, axis=-1), 
                        tf.gather_nd(labels, order_for_gather)], axis=-1)

    if visu:
        features_np = np.array(features)
        labels_np = np.array(labels)
        fig, axes = plt.subplots(nrows=1, ncols=k_max - k_min + 1, sharey=True, figsize=(5 * (k_max - k_min - 1), 3))

        X_coord, Y_coord = features_np[:, 0], features_np[:, 1]
        for i in range(k_max - k_min + 1):
            axes[i].set_title('Task {}'.format(i))
            pos = labels_np[:, i] == 1
            neg = labels_np[:, i] == 0
            axes[i].scatter(X_coord[pos], Y_coord[pos], s=0.5, color='lime')
            axes[i].scatter(X_coord[neg], Y_coord[neg], s=0.5, color='crimson')
        plt.show()

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batchsize))


SUPPORTED_PERMUTATION_TOYS = {"fractal_linsep": fractal_separation,
                              "fractal_linsep_maonet": fractal_separation_maonet,
                              "fractal_linsepv2": fractal_separationv2,
                              "D_separation": D_separation}

if __name__ == '__main__':
    D_separation(2, 10, 1, 2)
