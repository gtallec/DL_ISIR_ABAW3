import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

import os

from datasets.toys.config import SAVE_NORM_PATH
from datasets.toys.toy_utils import bernouilli


def corrupt_f(y, c):
    """
    Ins:
    y holds binary inputs.
    c (same shape as x) holds corruption indicator 1 means corruption 0 means x remains unchanged.
    Outs:
    y_c corrupted version of y.
    """
    return ((2 * y - 1) * (2 * c - 1) + 1)/2 


    return 
def corruption_chain_toy(N, p1, p2, mode='train', visu=False, batchsize=32, seed=None, meta=False):
    np.random.seed(seed)

    # (N, 2)
    X = 2 * np.random.random(size=(N, 2)) - 1

    # (N,)
    p1 = p1 * np.ones((N, ))
    # (N,)
    p2 = p2 * np.ones((N, ))

    # (N,)
    c1 = bernouilli(p1)
    # (N,)
    c2 = bernouilli(p2)
   
    # (N, )
    y = np.logical_and(0 <= X[:, 0], X[:, 0] <= 1).astype(float)
    y1 = corrupt_f(y, 1 - c1)
    y2 = corrupt_f(y1, 1 - c2)

    labels = np.stack([y1, y2], axis=-1) 

    save_file = os.path.join(SAVE_NORM_PATH, 'corruption_chain', 'normalisation.npz')
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

    if visu:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 5))

        X_coord, Y_coord = features[:, 0], features[:, 1]
        # First task is a K class segmentation
        for i in range(0, 2):
            axes[i].set_title('Task {}'.format(i))
            pos = labels[:, i] == 1
            neg = labels[:, i] == 0
            axes[i].scatter(X_coord[pos], Y_coord[pos], s=0.5, color='lime')
            axes[i].scatter(X_coord[neg], Y_coord[neg], s=0.5, color='crimson')
        plt.show()

    features = tf.dtypes.cast(features, tf.float32)
    labels = tf.dtypes.cast(labels, tf.float32)
    
    if not(meta):
        return (tf.data.Dataset.from_tensor_slices((features, labels))
                .shuffle(N)
                .batch(batchsize))
    else:
        return (tf.data.Dataset.from_tensor_slices(labels)
                .shuffle(N)
                .batch(batchsize))
       



SUPPORTED_MOMHA_TOYS = {"corruption_toy": corruption_chain_toy}

if __name__ == '__main__':
    corruption_chain_toy(N=500,
                         p1=0.5,
                         p2=0.7,
                         visu=True)
