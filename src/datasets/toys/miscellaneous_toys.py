import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

from datasets.toys.toy_utils import bernouilli, T_sigmoid
from datasets.toys.config import SAVE_NORM_PATH

def empty_ball(N, D, r_0, sigma, batch_size, mode='train', visu=False):
    X = 2 * np.random.random(size=(N, D)) - 1
    p_X_1 = np.exp(-(1/2)*(np.power((r_0 - np.linalg.norm(X, axis=1))
                                    /
                                    sigma, 2)))

    p_X_2 = np.exp(-(1/2)*(np.power((2*r_0 - np.linalg.norm(X, axis=1))
                                    /
                                    sigma, 2)))

    Y_1 = bernouilli(p_X_1)
    Y_2 = bernouilli(np.power(p_X_1, Y_1) * np.power(p_X_2, 1 - Y_1))
    eps = 1e-7


    save_file = os.path.join(SAVE_NORM_PATH, 'empty_ball', 'normalisation.npz')
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


    features = ((X - mean)
                /
                (std + eps))
    labels = np.zeros((N, 2))
    labels[:, 0] = Y_1
    labels[:, 1] = Y_2

    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        null_class = np.logical_and(Y_1 == 0, Y_2 == 0)
        first_class = np.logical_and(Y_1 == 1, Y_2 == 0)
        second_class = np.logical_and(Y_1 == 0, Y_2 == 1)
        last_class = np.logical_and(Y_1 == 1, Y_2 == 1)

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative', color='blue') 
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='class 1', color='red')
        plt.scatter(X_coord[second_class], Y_coord[second_class], label='class 2', color='yellow')
        plt.scatter(X_coord[last_class], Y_coord[last_class], label='class 1 and 2', color='pink')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def donuts(N, D, r, sigma, batch_size, stoc=False, mode='train', visu=False):
    X = 2 * np.random.random(size=(N, D)) - 1
    if stoc:
        p = np.exp(-(1/2)*(np.power((r - np.linalg.norm(X, axis=1))
                                    /
                                    sigma, 2)))
        Y_1 = bernouilli(p)
    else:
        distance_to_circle = np.power(r - np.linalg.norm(X, axis=1), 2)
        Y_1 = (distance_to_circle < np.power(sigma/2, 2)).astype(float)

    save_file = os.path.join(SAVE_NORM_PATH, 'donuts', 'normalisation.npz')
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

    print(mode, ' mean : ', np.mean(features, axis=0))
    print(mode, ' std : ', np.std(features, axis=0))
    labels = Y_1

    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        null_class = (Y_1 == 0)
        first_class = (Y_1 == 1)

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative', color='blue') 
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='class 1', color='red')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels[:, np.newaxis], dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))


def double_donuts(N, D, r, sigma, batch_size, stoc=False, mode='train', visu=False):
    X = 2 * np.random.random(size=(N, D)) - 1
    if stoc:
        p_X = np.exp(-(1/2)*(np.power((r - np.linalg.norm(X, axis=1))
                                      /
                                      sigma, 2)))
        Y_1 = bernouilli(p_X)
        Y_2 = bernouilli(Y_1 * p_X)
    else:
        distance_to_circle = np.power(r - np.linalg.norm(X, axis=1), 2)
        Y_1 = (distance_to_circle < (np.power(3 * sigma, 2))).astype(float)
        Y_2 = (distance_to_circle < (np.power(3 * sigma, 2))/(2)).astype(float)

    save_file = os.path.join(SAVE_NORM_PATH, 'double_donuts', 'normalisation.npz')
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

    print(mode, ' mean : ', np.mean(features, axis=0))
    print(mode, ' std : ', np.std(features, axis=0))
    labels = np.zeros((N, 2))
    labels[:, 0] = Y_1
    labels[:, 1] = Y_2

    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        null_class = np.logical_and((Y_1 == 0),
                                    (Y_2 == 0))
        first_class = np.logical_and((Y_1 == 0),
                                     (Y_2 == 1))
        second_class = np.logical_and((Y_1 == 1),
                                      (Y_2 == 0))
        third_class = np.logical_and((Y_1 == 1),
                                     (Y_2 == 1))

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative', color='blue') 
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='class 1', color='red')
        plt.scatter(X_coord[second_class], Y_coord[second_class], label='class 2', color='green')
        plt.scatter(X_coord[third_class], Y_coord[third_class], label='class 3', color='yellow')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))
def vertical_boundary(N, D, T, batch_size, mode='train', visu=False):
    X = 2 * np.random.random(size=(N, D)) - 1
    p = T_sigmoid(T)(X[:, 0])
    Y_1 = bernouilli(p)
    Y_2 = bernouilli(1-p)

    save_file = os.path.join(SAVE_NORM_PATH, 'vertical_boundary', 'normalisation.npz')
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
    labels = np.zeros((N, 2))
    labels[:, 0] = Y_1
    labels[:, 1] = Y_2

    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        null_class = np.logical_and(Y_1 == 0, Y_2 == 0)
        first_class = np.logical_and(Y_1 == 1, Y_2 == 0)
        second_class = np.logical_and(Y_1 == 0, Y_2 == 1)
        last_class = np.logical_and(Y_1 == 1, Y_2 == 1)

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative', color='blue') 
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='class 1', color='red')
        plt.scatter(X_coord[second_class], Y_coord[second_class], label='class 2', color='yellow')
        plt.scatter(X_coord[last_class], Y_coord[last_class], label='class 1 and 2', color='pink')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def strips(N, D, w, width, eps, batch_size, mode='train', visu=False): 
    X = 2 * np.random.random(size=(N, D)) - 1
    w = np.array(w).reshape(D, 1)
    w = w / np.linalg.norm(w)
    sum_w = np.sum(w)
    M = int(np.floor(2 * sum_w/width))
    b = -sum_w + width * np.arange(M + 1)
    hyperplane_distances = np.abs(np.tile(X @ w, reps=(1, M + 1)) - np.tile(b[np.newaxis, :], reps=(N, 1)))
    closest_hyperplane = np.min(hyperplane_distances, axis=1)
    Y_1 = (closest_hyperplane < eps).astype(float)


    save_file = os.path.join(SAVE_NORM_PATH, 'strips', 'normalisation.npz')
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
    labels = Y_1.reshape(N, 1)
    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        null_class = (Y_1 == 0)
        first_class = (Y_1 == 1)

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative example', color='blue')
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='posive example', color='red')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def demi_plane(N, D, mode='train', batch_size=32, visu=False):
    X = 2 * np.random.random(size=(N, D)) - 1
    Y_0 = X[:, 0] > 0
    print('Y_0.shape : ', Y_0.shape)
    Y_1 = X[:, 1] > 0
    Y_2 = X[:, 0] * X[:, 1] > 0

    save_file = os.path.join(SAVE_NORM_PATH, 'demi_plane', 'normalisation.npz')
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
    labels = np.zeros((N, 3))
    labels[:, 0] = Y_0
    labels[:, 1] = Y_1
    labels[:, 2] = Y_2
    
    if visu:

        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        plt.figure()
        plt.title("Task 0")
        task_0_pos = (Y_0 == 1)
        task_0_neg = (Y_0 == 0)

        plt.scatter(X_coord[task_0_pos], Y_coord[task_0_pos], color='green', label='positive examples')
        plt.scatter(X_coord[task_0_neg], Y_coord[task_0_neg], color='red', label='negative examples')
        plt.legend()

        plt.figure()
        plt.title("Task 1")
        task_1_pos = (Y_1 == 1)
        task_1_neg = (Y_1 == 0)

        plt.scatter(X_coord[task_1_pos], Y_coord[task_1_pos], color='green', label='positive examples')
        plt.scatter(X_coord[task_1_neg], Y_coord[task_1_neg], color='red', label='negative examples')
        plt.legend()

        plt.figure()
        plt.title("Task 2")
        task_2_pos = (Y_2 == 1)
        task_2_neg = (Y_2 == 0)

        plt.scatter(X_coord[task_2_pos], Y_coord[task_2_pos], color='green', label='positive examples')
        plt.scatter(X_coord[task_2_neg], Y_coord[task_2_neg], color='red', label='negative examples')
        plt.legend()

        only_zero = np.logical_and(Y_0 == 1,
                                   Y_1 == 0,
                                   Y_2 == 0)
        only_one = np.logical_and(Y_0 == 0,
                                  Y_1 == 1,
                                  Y_2 == 0)
        all_positive = np.logical_and(Y_0 == 1,
                                      Y_1 == 1,
                                      Y_2 == 1)

        all_negative = np.logical_and(Y_0 == 0,
                                      Y_1 == 0,
                                      Y_2 == 1)

        plt.figure()
        plt.scatter(X_coord[only_zero], Y_coord[only_zero], label='task 0 only', color='blue')
        plt.scatter(X_coord[only_one], Y_coord[only_one], label='task 1 only', color='yellow')
        plt.scatter(X_coord[all_positive], Y_coord[all_positive], label='task 1,2,3', color='red')
        plt.scatter(X_coord[all_negative], Y_coord[all_negative], label='task 3 only', color='green')
        plt.legend()
        plt.show()


    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def sinusoidal_separation(N, D, T, A, mode='train', batch_size=32, visu=False):
    X = 2 * np.random.random(size=(N, D)) - 1
    right = (X[:, 0] - A * np.sin(2 * np.pi * X[:, 1]/T))
    up = (X[:, 1] - A * np.sin(2 * np.pi * X[:, 0]/T))

    Y_1 = right > 0
    Y_2 = right * up > 0

    save_file = os.path.join(SAVE_NORM_PATH, 'sinusoidal_separation', 'normalisation.npz')
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
    labels = np.zeros((N, 2))
    labels[:, 0] = Y_1
    labels[:, 1] = Y_2

    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]


        plt.figure()
        plt.title('First Task')
        positive_class_1 = (Y_1 == 1)
        negative_class_1 = (Y_1 == 0)

        plt.scatter(X_coord[positive_class_1], Y_coord[positive_class_1], label='positive example', color='green')
        plt.scatter(X_coord[negative_class_1], Y_coord[negative_class_1], label='negative example', color='red') 
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('Second Task')
        positive_class_2 = (Y_2 == 1)
        negative_class_2 = (Y_2 == 0)

        plt.scatter(X_coord[positive_class_2], Y_coord[positive_class_2], label='positive example', color='green')
        plt.scatter(X_coord[negative_class_2], Y_coord[negative_class_2], label='negative example', color='red') 
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))


def opposite_strips_in_donuts(N, D, w, width, eps, r, sigma, batch_size, mode='train', visu=False, switch=False, seed=0):
    np.random.seed(seed)
    X = 2 * np.random.random(size=(N, D)) - 1


    distance_to_circle = np.power(r - np.linalg.norm(X, axis=1), 2)
    Y_1 = (distance_to_circle < np.power(sigma/2, 2)).astype(float)

    w = np.array(w).reshape(D, 1)
    w = w / np.linalg.norm(w)
    sum_w = np.sum(w)
    M = int(np.floor(2 * sum_w/width))
    b = -sum_w + width * np.arange(M + 1)
    hyperplane_distances = np.abs(np.tile(X @ w, reps=(1, M + 1)) - np.tile(b[np.newaxis, :], reps=(N, 1)))
    closest_hyperplane = np.min(hyperplane_distances, axis=1)
    
    Y_2 = ((2 * Y_1 - 1) * closest_hyperplane < (2 * Y_1 - 1) * eps).astype(float)


    save_file = os.path.join(SAVE_NORM_PATH, 'opposite_strips', 'normalisation.npz')
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
    labels = np.zeros((N, 2))
    labels[:, 0] = Y_1
    labels[:, 1] = Y_2

    if switch:
        labels[:, 0] = Y_2
        labels[:, 1] = Y_1


    if visu:

        X_coord = features[:, 0]
        Y_coord = features[:, 1]


        plt.figure()
        plt.title('First Task')
        positive_class_1 = (Y_1 == 1)
        negative_class_1 = (Y_1 == 0)

        plt.scatter(X_coord[positive_class_1], Y_coord[positive_class_1], label='positive example', color='green')
        plt.scatter(X_coord[negative_class_1], Y_coord[negative_class_1], label='negative example', color='red') 
        plt.legend()

        plt.figure()
        plt.title('Second Task')
        positive_class_2 = (Y_2 == 1)
        negative_class_2 = (Y_2 == 0)

        plt.scatter(X_coord[positive_class_2], Y_coord[positive_class_2], label='positive example', color='green')
        plt.scatter(X_coord[negative_class_2], Y_coord[negative_class_2], label='negative example', color='red') 
        plt.legend()

        plt.figure()
        null_class = np.logical_and(Y_1 == 0, Y_2 == 0)
        first_class = np.logical_and(Y_1 == 1, Y_2 == 0)
        second_class = np.logical_and(Y_1 == 0, Y_2 == 1)
        last_class = np.logical_and(Y_1 == 1, Y_2 == 1)

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative example', color='blue')
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='class 0', color='green')
        plt.scatter(X_coord[second_class], Y_coord[second_class], label='class 1', color='yellow')
        plt.scatter(X_coord[last_class], Y_coord[last_class], label='class 0 and 1', color='red')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def opposite_strips(N, D, w, width, eps, batch_size, mode='train', visu=False, switch=False, seed=0):
    np.random.seed(seed)
    X = 2 * np.random.random(size=(N, D)) - 1

    Y_1 = (X[:, 1] > 0).astype(float)

    w = np.array(w).reshape(D, 1)
    w = w / np.linalg.norm(w)
    sum_w = np.sum(w)
    M = int(np.floor(2 * sum_w/width))
    b = -sum_w + width * np.arange(M + 1)
    hyperplane_distances = np.abs(np.tile(X @ w, reps=(1, M + 1)) - np.tile(b[np.newaxis, :], reps=(N, 1)))
    closest_hyperplane = np.min(hyperplane_distances, axis=1)
    
    Y_2 = ((2 * Y_1 - 1) * closest_hyperplane < (2 * Y_1 - 1) * eps).astype(float)


    save_file = os.path.join(SAVE_NORM_PATH, 'opposite_strips', 'normalisation.npz')
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
    labels = np.zeros((N, 2))
    labels[:, 0] = Y_1
    labels[:, 1] = Y_2

    if switch:
        labels[:, 0] = Y_2
        labels[:, 1] = Y_1


    if visu:
        X_coord = features[:, 0]
        Y_coord = features[:, 1]

        null_class = np.logical_and(Y_1 == 0, Y_2 == 0)
        first_class = np.logical_and(Y_1 == 1, Y_2 == 0)
        second_class = np.logical_and(Y_1 == 0, Y_2 == 1)
        last_class = np.logical_and(Y_1 == 1, Y_2 == 1)

        plt.scatter(X_coord[null_class], Y_coord[null_class], label='negative example', color='blue')
        plt.scatter(X_coord[first_class], Y_coord[first_class], label='class 0', color='green')
        plt.scatter(X_coord[second_class], Y_coord[second_class], label='class 1', color='yellow')
        plt.scatter(X_coord[last_class], Y_coord[last_class], label='class 0 and 1', color='red')
        plt.legend()
        plt.show()

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def corrupted_chain(N, p, q, mode='train', batch_size=32):
    X = 2 * np.random.random(size=(N, 2)) - 1
    labels = np.zeros((N, 2))

    Y_1 = (X[:, 0] < -1 + 2 * p).astype(float)
    labels[:, 0] = Y_1

    b_q = (np.random.random(size=(N,)) - q <= 0).astype(float)
    Y_2 = Y_1 * b_q
    labels[:, 1] = Y_2

    fig, axes = plt.subplots(nrows=1, ncols=2)
    x, y = X[:, 0], X[:, 1]
    for i in range(2):
        pos = (labels[:, i] == 1)
        neg = (labels[:, i] == 0)
        axes[i].scatter(x[pos], y[pos], s=0.1, color='lime')
        axes[i].scatter(x[neg], y[neg], s=0.1, color='crimson')
    plt.show()


    save_file = os.path.join(SAVE_NORM_PATH, 'corrupted_chain', 'normalisation.npz')
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


    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))



def corruption_toy(N, n, p, mode='train', batch_size=32):
    X = 2 * np.random.random((N, 2)) - 1
    labels = np.concatenate([np.ones((N, 1)), np.zeros((N, n))], axis=1)

    for i in range(1, n + 1):
        toy_sign = np.prod(2 * labels[:, :i] - 1, axis=1)
        y_i_tilde = (toy_sign * X[:, 0] >= 0).astype(float)
        corruption = (np.random.random((N,)) - p <= 0).astype(float)
        labels[:, i] = (1/2) * ((2 * corruption - 1) * (2 * y_i_tilde - 1) + 1)

    fig, axes = plt.subplots(nrows=1, ncols=n)
    x, y = X[:, 0], X[:, 1]
    for i in range(1, n + 1):
        pos = (labels[:, i] == 1)
        neg = (labels[:, i] == 0)
        axes[i - 1].scatter(x[pos], y[pos], s=0.1, color='lime')
        axes[i - 1].scatter(x[neg], y[neg], s=0.1, color='crimson')


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
    labels = labels[:, 1:]


    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def corruption_toyv2(N, n, p_1, p_2, mode='train', batch_size=32):
    X = 2 * np.random.random((N, 2)) - 1
    q = p_2 / (1 + p_1 + p_2)

    labels = np.zeros((N, n))
    labels[:, 0] = (X[:, 0] < -1 + 2*q).astype(float)
    for i in range(1, n):
        Y_n_1 = labels[:, i-1]
        labels[:, i] = (np.random.random((N, )) - (Y_n_1 * p_1 + (1 - Y_n_1) * p_2) <= 0).astype(float)

    """
    x, y = X[:, 0], X[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=n)
    for i in range(n):
        pos = (labels[:, i] == 1)
        neg = (labels[:, i] == 0)
        axes[i].scatter(x[pos], y[pos], s=0.5, color='lime')
        axes[i].scatter(x[neg], y[neg], s=0.3, color='crimson')
    plt.show()
    """


    save_file = os.path.join(SAVE_NORM_PATH, 'corruption_toyv2', 'normalisation.npz')
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

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def corruption_toyv3(N, n, q, p_1, p_2, mode='train', batch_size=32):
    X = 2 * np.random.random((N, 2)) - 1
    Y = np.zeros((N, n))

    Y[:, 0] = (X[:, 0] < -1 + 2*q).astype(float)
    Y_tilde = Y[:, 0]
    for i in range(1, n):
        Y[:, i] = (np.random.random((N, )) - (Y_tilde * p_1 + (1 - Y_tilde) * p_2) <= 0).astype(float)
        Y_tilde = np.logical_or(Y_tilde, Y[:, i])


    x, y = X[:, 0], X[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=n)
    for i in range(n):
        pos = (Y[:, i] == 1)
        neg = (Y[:, i] == 0)
        axes[i].scatter(x[pos], y[pos], s=0.5, color='lime')
        axes[i].scatter(x[neg], y[neg], s=0.3, color='crimson')
    plt.show()

    save_file = os.path.join(SAVE_NORM_PATH, 'corruption_toyv3', 'normalisation.npz')
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

    features = tf.constant(features, dtype=tf.float32)
    Y = tf.constant(Y, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, Y))
            .shuffle(N)
            .batch(batch_size))


def corruption_toyv4(N, n, q, p_1, p_2, mode='train', batch_size=32):
    X = 2 * np.random.random((N, 2)) - 1
    Y = np.zeros((N, n))

    Y[:, 0] = (X[:, 0] < -1 + 2*q).astype(float)
    Y_tilde = Y[:, 0]
    for i in range(1, n):
        Y[:, i] = (np.random.random((N, )) - (Y_tilde * p_1 + (1 - Y_tilde) * p_2) <= 0).astype(float)
        Y_tilde = (Y_tilde + Y[:, i]) % 2


    x, y = X[:, 0], X[:, 1]
    fig, axes = plt.subplots(nrows=1, ncols=n, sharey=True, figsize=(5 * n, 5))
    for i in range(n):
        pos = (Y[:, i] == 1)
        neg = (Y[:, i] == 0)
        axes[i].scatter(x[pos], y[pos], s=0.5, color='lime')
        axes[i].scatter(x[neg], y[neg], s=0.3, color='crimson')
    plt.show()

    save_file = os.path.join(SAVE_NORM_PATH, 'corruption_toyv4', 'normalisation.npz')
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

    features = tf.constant(features, dtype=tf.float32)
    Y = tf.constant(Y, dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, Y))
            .shuffle(N)
            .batch(batch_size))



def circular_toy(N, n, T=1, mode='train', batch_size=32):
    X = 2 * np.random.random((N, 2)) - 1
    X_norm = X / np.tile(np.linalg.norm(X, axis=1)[:, np.newaxis], reps=(1, 2))
    labels = np.concatenate([np.ones((N, 1))/2, np.zeros((N, n))], axis=1)

    for i in range(1, n + 1):
        angle = np.sum((2 * labels[:, :i] - 1) * np.arange(i), axis=1)

        w_x = np.cos(angle * 2 * np.pi / n)[:, np.newaxis]
        w_y = np.sin(angle * 2 * np.pi / n)[:, np.newaxis]
        W = np.concatenate([w_x, w_y], axis=1)

        p_i = T_sigmoid(T)(np.sum(W * X_norm, axis=1))
        labels[:, i] = (np.random.random(size=(N,)) - p_i <= 0).astype(float)

        x, y = X[:, 0], X[:, 1]

    """
    fig, axes = plt.subplots(nrows=1, ncols=n)
    for i in range(1, n + 1):
        pos = (labels[:, i] == 1)
        neg = (labels[:, i] == 0)
        axes[i - 1].scatter(x[pos], y[pos], s=0.1, color='lime')
        axes[i - 1].scatter(x[neg], y[neg], s=0.1, color='crimson')
    plt.show()
    """

    save_file = os.path.join(SAVE_NORM_PATH, 'circular_toy', 'normalisation.npz')
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

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels[:, 1:], dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))

def binary_canal(N, p, p_i, mode='train', batch_size=32):
    X = 2 * np.random.random((N, 2)) - 1
    x = X[:, 0]
    y = X[:, 1]
    n = len(p_i)
    x_tiled = np.tile(x[:, np.newaxis], reps=(1, n))
    thresholds = np.tile((-1 + 2 * np.array(p_i))[np.newaxis, :], reps=(N, 1))

    uncorrupted_labels = 2 * ((x_tiled <= thresholds).astype(float)) - 1
    canal_noise = np.tile(2 * ((np.random.random(size=(N, 1)) - p <= 0).astype(float)) - 1, reps=(1, n))
    labels = (uncorrupted_labels * canal_noise + 1)/2


    fig, axes = plt.subplots(nrows=1, ncols=n)
    for i in range(n):
        pos = (labels[:, i] == 1)
        neg = (labels[:, i] == 0)
        axes[i].scatter(x[pos], y[pos], s=0.3, color='lime')
        axes[i].scatter(x[neg], y[neg], s=0.1, color='crimson')
    plt.show()

    save_file = os.path.join(SAVE_NORM_PATH, 'binary_canal', 'normalisation.npz')
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

    features = tf.constant(features, dtype=tf.float32)
    labels = tf.constant(labels[:, 1:], dtype=tf.float32)

    return (tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(N)
            .batch(batch_size))




SUPPORTED_MISCELLANEOUS_TOYS = {"empty_ball": empty_ball,
                                "vertical_boundary": vertical_boundary,
                                "donuts": donuts,
                                "double_donuts": double_donuts,
                                "strips": strips,
                                "opposite_strips": opposite_strips,
                                "opposite_strips_in_donuts": opposite_strips_in_donuts,
                                "demi_plane": demi_plane,
                                "circular_separation": circular_toy,
                                "corruption_toyv2": corruption_toyv2,
                                "corruption_toyv3": corruption_toyv3,
                                "corrupted_chain": corrupted_chain,
                                "binary_canal": binary_canal}


