import tensorflow as tf
import os

from datasets.disfa.config import PREPROCESSED_META, PREPROCESSED_VIDEO_TEMPLATE, FRAME_TEMPLATE, SOTA_PROJECTION, JAANET_path
from datasets.disfa.data_augmentation import data_augmentation


def preprocess_images(images, labels):
    return tf.image.per_image_standardization(2 * images - 1), labels

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

def filter_fold(folds):
    folds = tf.constant(folds)

    def fun(*args):
        fold = args[-1]
        return tf.math.reduce_prod(fold - folds) == 0
    return fun

def label_projection(*args):
    label = args[:-1]
    return label

def load_image(*args):
    labels = args[:-1]
    path = args[-1]
    return decode_img(path), tf.stack(labels, axis=0)

def intensity_to_activation(images, labels):
    return images, tf.dtypes.cast(labels >= 2, dtype=tf.float32)

def decode_img(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def load_dataset_meta(folds, assignment_type='jaanet'):
    folds = sorted(folds)
    folds = [str(fold) for fold in folds]
    if assignment_type == 'jaanet':
        str_fold = ''.join(folds)
        dataset_file = os.path.join(JAANET_path, 'fold{}.csv'.format(str_fold))
        dataset_types = 12 * [float()] + [str()]
        return (tf.data.experimental.CsvDataset(dataset_file,
                                                record_defaults=dataset_types,
                                                header=True)
                .cache())
        
    dataset_file = PREPROCESSED_META.format(assignment_type)
    dataset_types = [str()] + 12 * [float()] + [str(), int()]
    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .filter(filter_fold(folds))
            .cache())

def load_subdisfa_meta(folds, mode):
    folds = sorted(folds)
    folds = [str(fold) for fold in folds]
    dataset_file = os.path.join(JAANET_path, 'hypertuning', 'fold{}_{}.csv'.format(''.join(folds), mode))
    dataset_types = 12 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .cache())

def gen_subdisfa(folds, n_au, batchsize, mode, meta=False, augmentation=True, intensity=False):
    meta_dataset = load_subdisfa_meta(folds, mode)
    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    projection = SOTA_PROJECTION[:n_au] 
    if meta:
        meta_dataset = (meta_dataset
                        .map(label_projection, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize))
        return meta_dataset
    else:
        loaded_dataset = (meta_dataset.shuffle(count)
                          .map(load_image, tf.data.experimental.AUTOTUNE))
        if augmentation and mode == 'train':
            print("Data Augmentation")
            loaded_dataset = loaded_dataset.map(data_augmentation(), tf.data.experimental.AUTOTUNE)

        load_dataset = (loaded_dataset.batch(batchsize)
                        .map(preprocess_images, tf.data.experimental.AUTOTUNE)
                        .map(intensity_to_activation, tf.data.experimental.AUTOTUNE)
                        .map(projective_map(projection), tf.data.experimental.AUTOTUNE)
                        .prefetch(tf.data.experimental.AUTOTUNE))

        return load_dataset



def gen_disfa(fold, n_au, batchsize, mode, fold_assignement='jaanet', meta=False, augmentation=False, intensity=False):
    meta_dataset = load_dataset_meta(fold, assignment_type=fold_assignement)
    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    projection = SOTA_PROJECTION[:n_au] 
    if meta:
        meta_dataset = (meta_dataset
                        .map(label_projection, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize))
        return meta_dataset
    else:
        loaded_dataset = (meta_dataset.shuffle(count)
                          .map(load_image, tf.data.experimental.AUTOTUNE))
        if augmentation and mode == 'train':
            print("Data Augmentation")
            loaded_dataset = loaded_dataset.map(data_augmentation(), tf.data.experimental.AUTOTUNE)

        load_dataset = (loaded_dataset.batch(batchsize)
                        .map(preprocess_images, tf.data.experimental.AUTOTUNE)
                        .map(intensity_to_activation, tf.data.experimental.AUTOTUNE)
                        .map(projective_map(projection), tf.data.experimental.AUTOTUNE)
                        .prefetch(tf.data.experimental.AUTOTUNE))

        return load_dataset


if __name__ == '__main__':
    disfa = gen_disfa([2], 8, 32, 'train', 'jaanet')
