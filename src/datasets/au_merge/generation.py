import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from datasets.abaw3.config import CSV_TEMPLATE, SANITY_VIDEO_CSV_TEMPLATE, VIDEO_CSV_TEMPLATE, VIDEO_SUMMARY_TEMPLATE, VIDEO_TENSOR_TEMPLATE
from datasets.bp4d.config import BP4D_FOR_ABAW3
from datasets.disfa.config import DISFA_FOR_ABAW3
from datasets.abaw3.config import TRAIN_ABAW3, VALID_ABAW3, TRAINVAL_ABAW3
from datasets.au_merge.config import BP4D_DISFA_CSV_TEMPLATE


def decode_image(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, size=(299, 299))

def decode_labels(labels):
    serialized_tensor = tf.io.read_file(labels)
    return tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

def decode_paths(labels_path, img_path):
    return decode_image(img_path), decode_labels(labels_path) 

def drop_image(labels_path, img_path):
    return labels_path

def load_dataset_meta(csv_path):
    dataset_types = [str(), str()]
    return tf.data.experimental.CsvDataset(csv_path,
                                           record_defaults=dataset_types,
                                           header=True)

def gen_au_merge(mode, batchsize, abaw3_prop=1.0, disfa_prop=0.0, bp4d_prop=0.0, meta=False, subsample=1):
    if mode == 'train' or mode == 'trainval':
        if mode == 'train':
            ABAW3_FOR_ABAW3 = TRAIN_ABAW3
        else:
            ABAW3_FOR_ABAW3 = TRAINVAL_ABAW3

        loaded_dataset_files = [load_dataset_meta(ABAW3_FOR_ABAW3),
                                load_dataset_meta(DISFA_FOR_ABAW3),
                                load_dataset_meta(BP4D_FOR_ABAW3)]
        sampling_proportions = [abaw3_prop, disfa_prop, bp4d_prop]
        meta_dataset = (tf.data.Dataset.sample_from_datasets(loaded_dataset_files,
                                                             weights=sampling_proportions,
                                                             stop_on_empty_dataset=True)
                        .shard(subsample, 0))
    elif mode == 'valid':
        ABAW3_FOR_ABAW3 = VALID_ABAW3
        meta_dataset = (load_dataset_meta(ABAW3_FOR_ABAW3)
                        .shard(subsample, 0))
    else:
        sys.exit('unsupported mode : {}'.format(mode))

    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    if meta:
        meta_dataset = (meta_dataset
                        .map(drop_image, tf.data.experimental.AUTOTUNE)
                        .map(decode_labels, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize)
                        .prefetch(tf.data.experimental.AUTOTUNE))
        return meta_dataset
    else:
        loaded_dataset = (meta_dataset
                          .shuffle(count)
                          .map(decode_paths, tf.data.experimental.AUTOTUNE)
                          .batch(batchsize)
                          .prefetch(tf.data.experimental.AUTOTUNE))
        
        return loaded_dataset

def gen_bp4d_disfa_meta(mode, subsample):
    dataset_types = [str(), str()]
    return (tf.data.experimental.CsvDataset(filenames=BP4D_DISFA_CSV_TEMPLATE.format(mode),
                                            record_defaults=dataset_types,
                                            header=True)
            .shard(subsample, 0)
            .cache())

def gen_bp4d_disfa(mode, batchsize, meta=False, subsample=1):
    meta_dataset = gen_bp4d_disfa_meta(mode, subsample=subsample)
    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    if meta:
        meta_dataset = (meta_dataset
                        .map(drop_image, tf.data.experimental.AUTOTUNE)
                        .map(decode_labels, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize)
                        .prefetch(tf.data.experimental.AUTOTUNE))
        return meta_dataset
    else:
        loaded_dataset = (meta_dataset
                          .shuffle(count)
                          .map(decode_paths, tf.data.experimental.AUTOTUNE)
                          .batch(batchsize)
                          .prefetch(tf.data.experimental.AUTOTUNE))
        
        return loaded_dataset



