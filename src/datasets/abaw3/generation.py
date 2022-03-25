import tensorflow as tf
import numpy as np
import pandas as pd
from datasets.abaw3.config import CSV_TEMPLATE, SANITY_VIDEO_CSV_TEMPLATE, VIDEO_CSV_TEMPLATE, VIDEO_SUMMARY_TEMPLATE, VIDEO_TENSOR_TEMPLATE

def separate_paths(paths):
    labels_path = paths[0]
    img_path = paths[-1]
    return labels_path, img_path

def concat_paths(*args):
    return tf.stack(args, axis=0)

def decode_image(img):
    img = tf.image.decode_jpeg(tf.io.read_file(img), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def decode_image_and_keep_path(path):
    return path, decode_image(path)

def decode_labels(labels):
    serialized_tensor = tf.io.read_file(labels)
    return tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)


def decode_tensor(path, dtype):
    serialized_tensor = tf.io.read_file(path)
    return tf.io.parse_tensor(serialized_tensor, out_type=dtype)

def decode_tensors(paths):
    labels_path = paths[0]
    img_path = paths[1] 
    return decode_image(img_path), decode_labels(labels_path)

def decode_paths(labels_path, img_path):
    return decode_image(img_path), decode_labels(labels_path) 

def drop_image(labels_path, img_path):
    return labels_path

def drop_labels(labels_path, img_path):
    return img_path

def load_dataset_meta(mode, subsample):
    dataset_types = [str(), str()]
    dataset_file = CSV_TEMPLATE.format(mode)

    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .shard(subsample, 0)
            .cache())

def gen_abaw3(mode, batchsize, meta=False, subsample=1):
    meta_dataset = load_dataset_meta(mode=mode, subsample=subsample)
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

def gen_test_abaw3(video, batchsize):
    video_dfs = pd.read_csv(CSV_TEMPLATE.format('test'))
    video_frames = video_dfs[video_dfs['video'] == video].sort_values(by='frame')
    video_df_numpy = np.squeeze(video_frames[['frame_path']].to_numpy(), axis=-1)
    return (tf.data.Dataset.from_tensor_slices(video_df_numpy)
            .map(decode_image_and_keep_path, tf.data.AUTOTUNE)
            .batch(batchsize)
            .prefetch(tf.data.AUTOTUNE))

def gen_valid_abaw3(video, batchsize):
    video_tensor_path = VIDEO_TENSOR_TEMPLATE.format(video)
    video_tensor = decode_tensor(video_tensor_path, dtype=tf.string)
    return (tf.data.Dataset.from_tensor_slices(video_tensor)
            .map(separate_paths, tf.data.AUTOTUNE)
            .map(drop_labels, tf.data.AUTOTUNE)
            .map(decode_image_and_keep_path, tf.data.AUTOTUNE)
            .batch(batchsize)
            .prefetch(tf.data.AUTOTUNE))
