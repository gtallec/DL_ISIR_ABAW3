from datasets.abaw2.config import ABAW2_PATH
import tensorflow as tf
import os

def load_dataset_meta(mode):
    dataset_path = os.path.join(ABAW2_PATH, '{}.csv'.format(mode))
    dataset_types = 12 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_path,
                                        record_defaults=dataset_types,
                                        header=True))

def count_labels(state, x):
    tf.print(state)
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
    return tf.image.per_image_standardization(2 * images - 1), labels

def gen_abaw2(batchsize, mode, n_au=12, meta=False, augmentation=False, subsample=1):
    meta_dataset = (load_dataset_meta(mode)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .cache())
    if mode == 'train':
        count = 1397513
    elif mode == 'valid':
        count = 445846
    
    if meta:
        return meta_dataset
    
    if mode == 'train':
        return (meta_dataset
                .shard(subsample, 0)
                .shuffle(count // subsample)
                .map(load_image_from_path)
                .batch(batchsize)
                .map(preprocess, tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE))
    
    return (meta_dataset
            .shard(subsample, 0)
            .shuffle(count//subsample)
            .map(load_image_from_path)
            .batch(batchsize)
            .map(preprocess, tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE))

