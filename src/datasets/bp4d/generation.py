import tensorflow as tf
import os

from datasets.bp4d.config import BP4D_PATH, FOLD_TEMPLATE
from datasets.bp4d.config import AU_COLUMNS, AU_BINARY, SEX_COLUMNS, ETHNICITY_COLUMNS, TASK_COLUMNS
from datasets.bp4d.data_augmentation import data_augmentation


def separate_path_from_label(*args):
    labels = args[:-1]
    path = args[-1]
    return path, tf.stack(labels, axis=0)

def decode_image(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def load_image(path, labels):
    return decode_image(path), labels

def drop_path(path, labels):
    return labels

def load_dataset_meta(fold, labels):
    dataset_types = []
    select_cols = []

    if 'AU_binary' in labels:
        dataset_types = 12 * [float()] + [str()]
        select_cols = AU_BINARY + [38] 
    else:
        if 'AU' in labels:
            dataset_types += 24 * [float()]
            select_cols += AU_COLUMNS
        
        if 'SEX' in labels:
            dataset_types += 2 * [float()]
            select_cols += SEX_COLUMNS

        if 'ETH' in labels:
            dataset_types += 4 * [float()]
            select_cols += ETHNICITY_COLUMNS

        if 'TASK' in labels:
            dataset_types += 8 * [float()]
            select_cols += TASK_COLUMNS

        dataset_types += [str()]
        select_cols += [38]

    dataset_file = FOLD_TEMPLATE.format(fold)
    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            select_cols=select_cols,
                                            header=True)
            .map(separate_path_from_label, tf.data.experimental.AUTOTUNE) 
            .cache())

def gen_bp4d(fold, labels, batchsize, meta=False, subsample=1):
    meta_dataset = load_dataset_meta(fold=fold, labels=labels).shard(subsample, 0)
    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    if meta:
        meta_dataset = (meta_dataset
                        .map(drop_path, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize)
                        .prefetch(tf.data.experimental.AUTOTUNE))
        return meta_dataset
    else:
        loaded_dataset = (meta_dataset
                          .shuffle(count)
                          .map(load_image, tf.data.experimental.AUTOTUNE)
                          .batch(batchsize)
                          .prefetch(tf.data.experimental.AUTOTUNE))
        
        return loaded_dataset
