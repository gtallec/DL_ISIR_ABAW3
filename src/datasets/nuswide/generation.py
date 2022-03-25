from datasets.nuswide.config import NUSWIDE_PATH
import tensorflow as tf
import os


def load_dataset_meta(mode):
    dataset_path = os.path.join(NUSWIDE_PATH, '{}.csv'.format(mode))
    dataset_types = 81 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_path,
                                            record_defaults=dataset_types,
                                            header=True))
def count_labels(state, x):
    return state + 1

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

def preprocess(images, labels):
    return tf.keras.applications.vgg16.preprocess_input(255 * images), labels


def gen_nuswide(batchsize, mode, meta=False, augmentation=False):
    meta_dataset = (load_dataset_meta(mode)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .cache())

    count = meta_dataset.reduce(tf.constant(0, dtype=tf.int64), count_labels)
    if meta:
        return meta_dataset

    return (meta_dataset
            .shuffle(count)
            .map(load_image_from_path)
            .batch(batchsize)
            .map(preprocess, tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE))





