from datasets.abaw2.config import ABAW2_PATH, PREPROCESSED_FOLD_TEMPLATE
import tensorflow as tf
import os

def load_dataset_meta(mode, label, debug, subsample):
    dataset_path = PREPROCESSED_FOLD_TEMPLATE.format(mode, label)
    if debug:
        basename, extension = os.path.splitext(dataset_path)
        dataset_path = basename + '_debug' + extension
    elif subsample:
        basename, extension = os.path.splitext(dataset_path)
        dataset_path = basename + '_subsample' + extension
    dataset_types = []
    if 'au' in label:
        dataset_types += 12 * [float()]
    if 'expr' in label:
        dataset_types += 7 * [float()]
    dataset_types += [str()]
    return (tf.data.experimental.CsvDataset(dataset_path,
                                            record_defaults=dataset_types,
                                            header=True))

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


"""
def expand_au(images, labels):
    # au_labels : (B, 12), emotion_labels : (B, 7)
    B = tf.shape(images)[0]
    au_labels, emotion_labels = labels[:, :12], labels[:, 12:]
    # 0 is for unlabels, 1 is for negative and 2 is for positive.
    # (B, 12)
    au_labels = tf.dtypes.cast(au_labels + 1, dtype=tf.int32)
    au_encoding = tf.constant([[-1, -1],
                               [1, 0],
                               [0, 1]], dtype=tf.float32)
    au_encoded_labels = tf.reshape(tf.gather(au_encoding,
                                             au_labels,
                                             axis=0),
                                   (B, 24))

    return images, tf.concat([au_encoded_labels, emotion_labels], axis=-1) 
"""

def gen_abaw2(batchsize, mode, label, debug=False, subsample=True, meta=False):
    meta_dataset = (load_dataset_meta(mode, label, debug=debug, subsample=subsample)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .cache())

    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)

    if meta:
        return (meta_dataset
                .map(drop_image, tf.data.experimental.AUTOTUNE)
                .batch(batchsize)
                .prefetch(tf.data.experimental.AUTOTUNE))
    
    return (meta_dataset
            .shuffle(count)
            .map(load_image_from_path, tf.data.experimental.AUTOTUNE)
            .batch(batchsize)
            .prefetch(tf.data.experimental.AUTOTUNE))
