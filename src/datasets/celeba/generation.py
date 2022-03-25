import tensorflow as tf

from datasets.celeba.config import PREPROCESSED_META_TEMPLATE, PREPROCESSED_IMAGES, get_projection 
from datasets.celeba.data_augmentation import data_augmentation

def decode_img(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def preprocess_image(img):
    return 2 * img - 1

def projective_map(projection, meta=False):
    tensor_projection = tf.constant(projection, dtype=tf.int32)
    if meta:
        def project(labels):
            projected_labels = tf.gather(labels,
                                         tensor_projection,
                                         axis=-1)
            return projected_labels
    else: 
        def project(features, labels):
            projected_labels = tf.gather(labels,
                                         tensor_projection,
                                         axis=-1)
            return features, projected_labels

    return project

def label_projection(images, labels):
    return labels

def separating_map(*args):
    return args[0], tf.stack(args[1:])

def load_image(img, label):
    return decode_img(tf.strings.join([PREPROCESSED_IMAGES, img],
                                      separator='/')), label

def preprocess(img, label):
    return preprocess_image(img), label

def load_dataset_meta(mode):
    dataset_meta = PREPROCESSED_META_TEMPLATE.format(mode)
    dataset_types = [str()] + 40 * [float()]
    return tf.data.experimental.CsvDataset(dataset_meta,
                                           record_defaults=dataset_types,
                                           header=True)

def count(state, x):
    return state + 1


def gen_celeba(batchsize, mode, meta=False, subsample=1, projection_mode='all'):
    dataset_meta = load_dataset_meta(mode).shard(subsample, 0)
    n_shuffle = dataset_meta.reduce(tf.constant(0, dtype=tf.int64), count)
    
    projection = get_projection(projection_mode)
    if meta:
        return (dataset_meta
                .map(separating_map, tf.data.experimental.AUTOTUNE)
                .map(label_projection, tf.data.experimental.AUTOTUNE)
                .batch(batchsize)
                .map(projective_map(projection, meta=True), tf.data.experimental.AUTOTUNE))
    else:
        dataset = (dataset_meta
                   .shuffle(n_shuffle)
                   .map(separating_map, tf.data.experimental.AUTOTUNE)
                   .map(load_image, tf.data.experimental.AUTOTUNE)
                   .batch(batchsize)
                   .map(projective_map(projection, meta=False), tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        return dataset 
