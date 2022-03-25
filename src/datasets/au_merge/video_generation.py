import tensorflow as tf
import numpy as np
from datasets.abaw3.config import VIDEO_SUMMARY_TEMPLATE, VIDEO_TENSOR_TEMPLATE, LABEL_PADDING, IMG_PADDING
from datasets.abaw3.generation import gen_abaw3

def separate_paths(paths):
    labels_path = paths[0]
    img_path = paths[-1]
    return img_path, labels_path

def concat_paths(*args):
    return tf.stack(args, axis=0)

def decode_image(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def decode_tensor(path, dtype): 
    serialized_tensor = tf.io.read_file(path)
    return tf.io.parse_tensor(serialized_tensor, out_type=dtype)

def decode_paths(img_path, labels_path):
    return decode_image(img_path), decode_tensor(labels_path, dtype=tf.float32)

def pad_tensor(tensor, window_size):
    shape = tf.shape(tensor)
    N = shape[0]
    remainder = N % window_size
    padding_tensor = tf.constant([LABEL_PADDING, IMG_PADDING], dtype=tf.string)[tf.newaxis, :]
    padding_tensor = tf.tile(padding_tensor, (window_size - remainder, 1))
    return tf.concat([tensor, padding_tensor],
                     axis=0)

def drop_image(labels_path, img_path):
    return labels_path

def load_video_dataset_meta(mode, S, subsample=1):
    mode_video_summary = np.load(VIDEO_SUMMARY_TEMPLATE.format(mode), allow_pickle=True)
    video_summary = [VIDEO_TENSOR_TEMPLATE.format(video) for video in mode_video_summary]
    video_summary_dataset = tf.data.Dataset.from_tensor_slices(video_summary)
    return (video_summary_dataset.interleave(lambda x: (tf.data.Dataset.from_tensor_slices(pad_tensor(tensor=decode_tensor(x, dtype=tf.string),
                                                                                                      window_size=S))
                                                        .window(size=S, drop_remainder=False)))
            .shard(subsample, 0)
            .cache())


def gen_abaw3_video(mode, batchsize, S, meta=False, sanity_subsample=1, aug_subsample=1):
    if meta:
        return gen_abaw3(mode=mode, batchsize=batchsize, meta=True, subsample=sanity_subsample)
    else:
        window_dataset = load_video_dataset_meta(mode=mode,
                                                 S=S,
                                                 subsample=sanity_subsample)
        n_windows = window_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
        dataset = (window_dataset
                   .shuffle(n_windows)
                   .shard(aug_subsample, 0)
                   .flat_map(lambda x: x)
                   .map(separate_paths, tf.data.AUTOTUNE)
                   .map(decode_paths, tf.data.AUTOTUNE)
                   .batch(S)
                   .batch(batchsize)
                   .prefetch(tf.data.AUTOTUNE))
    return dataset

