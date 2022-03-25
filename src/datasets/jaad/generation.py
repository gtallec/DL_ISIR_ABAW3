import os
import pandas as pd
import tensorflow as tf
from datasets.jaad.config import get_projection, JAAD_PATH 

def separate_image_from_label(*args):
    labels = args[:-1]
    path = args[-1]
    return path, tf.stack(labels, axis=0)

def decode_img(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def load_image_from_path(path, labels):
    return decode_img(path), labels

def vgg_preprocessing(images, labels):
    return tf.keras.applications.vgg16.preprocess_input(255 * images), labels

def count_labels(state, x):
    return state + 1

def load_dataset_meta(mode, projection, cache):
    projection_list = get_projection(projection)
    cache_folder = os.path.join(cache, mode)
    cache_file = os.path.join(cache, mode, projection + '.csv')
    if not(os.path.exists(cache_folder)):
        os.makedirs(cache_folder)

    if not(os.path.exists(cache_file)):
        df = pd.read_csv(os.path.join(JAAD_PATH, mode + '.csv'))
        df = df[projection_list + ['path']]
        df.to_csv(cache_file, index=False)

    dataset_types = (len(projection_list) * [float()]
                     + [str()])
    return (tf.data.experimental.CsvDataset(cache_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .cache())


def gen_jaad(batchsize, mode, projection='gait_attention', meta=False, augmentation=False, cache='../.cache'):
    meta_dataset = load_dataset_meta(mode, projection, cache).map(separate_image_from_label,
                                                                  tf.data.experimental.AUTOTUNE)

    count = meta_dataset.reduce(tf.constant(0, dtype=tf.int64), count_labels)
    if meta:
        return meta_dataset

    return (meta_dataset
            .shuffle(count)
            .map(load_image_from_path, tf.data.experimental.AUTOTUNE)
            .map(vgg_preprocessing, tf.data.experimental.AUTOTUNE)
            .batch(batchsize))


