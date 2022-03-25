import tensorflow as tf
import os
import pandas as pd

from datasets.dysfer.config import DYSFER_PATH, DYSFER_PROJECTION
from datasets.dysfer.data_augmentation import resample

def projective_map(projection, meta=False):
    tensor_projection = tf.constant(projection, dtype=tf.int32)

    def project_fl(features, labels):
        projected_labels = tf.gather(labels,
                                     tensor_projection,
                                     axis=-1)
        return features, projected_labels

    def project_l(labels):
        return tf.gather(labels, tensor_projection, axis=-1)

    if not(meta):
        return project_fl
    else:
        return project_l

def separate_image_from_label(*args):
    labels = args[:-1]
    path = args[-1]
    return path, tf.stack(labels, axis=0)

def drop_image(path, labels):
    return labels

def decode_img(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def load_image_from_path(path, labels):
    return decode_img(path), labels

def intensity_to_activation(threshold):
    def fun(images, labels):
        return images, tf.dtypes.cast(labels >= threshold, dtype=tf.float32)
    return fun


def load_dataset_meta(fold):
    # folds_str = ''.join([str(fold) for fold in sorted(folds)])
    dataset_file = os.path.join(DYSFER_PATH, 'preprocessed', 'folds', '{}.csv'.format(fold))
    dataset_types = 5 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .cache())

def gen_dysfer(batchsize, fold, threshold=2, subsample=1, meta=False, mix12=False):
    meta_dataset = (load_dataset_meta(fold)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .map(intensity_to_activation(threshold), tf.data.experimental.AUTOTUNE)
                    .shard(subsample, 0))
    if mix12:
        projection = [4, 2, 3]
    else:
        projection = [0, 1, 2, 3, 4]

    count = meta_dataset.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1)
    if meta:
        return (meta_dataset
                .map(drop_image, tf.data.experimental.AUTOTUNE)
                .batch(batchsize)
                .map(projective_map(projection, meta=meta), tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE))
    
    return (meta_dataset
            .shuffle(tf.dtypes.cast(count, tf.int64))
            .map(load_image_from_path, tf.data.experimental.AUTOTUNE)
            .batch(batchsize)
            .map(projective_map(projection, meta=meta), tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE))
