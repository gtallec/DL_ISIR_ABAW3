from datasets.peta.config import PETA_PATH, get_projection
import tensorflow as tf
import os


def load_dataset_meta(mode, partition):
    dataset_path = os.path.join(PETA_PATH,
                                'interfaces',
                                'partition_{}'.format(partition),
                                '{}.csv'.format(mode))
    dataset_types = 40 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_path,
                                            record_defaults=dataset_types,
                                            header=True))
def count_labels(state, x):
    return state + 1

def projective_map(projection):
    def fun(images, labels):
        return images, tf.gather(labels, projection, axis=1)
    return fun

def separate_image_from_label(*args):
    labels = args[:-1]
    path = args[-1]
    return path, tf.stack(labels, axis=0)

def decode_img(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def preprocess(images, labels):
    return 255 * images, labels

def load_image_from_path(path, labels):
    return decode_img(path), labels


def gen_peta(batchsize, mode, partition, meta=False, augmentation=False, projection_mode='sota'):
    meta_dataset = (load_dataset_meta(mode=mode, partition=partition)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .cache())
    projection = get_projection(projection_mode)

    count = meta_dataset.reduce(tf.constant(0, dtype=tf.int64), count_labels)
    if meta:
        return meta_dataset

    return (meta_dataset
            .shuffle(count)
            .map(load_image_from_path)
            .batch(batchsize)
            .map(preprocess, tf.data.experimental.AUTOTUNE)
            .map(projective_map(projection), tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE))

