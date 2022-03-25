import tensorflow as tf
import os
import numpy as np

from datasets.disfa.config import PREPROCESSED_META, SOTA_PROJECTION, JAANET_path, FOLD_TEMPLATE, fold_repartition_list, FRAMES_PER_VIDEO, DISFA, get_projection


def projective_map(projection, meta=False):
    tensor_projection = tf.constant(projection, dtype=tf.int32)

    def project_fl(features, labels):
        projected_labels = tf.gather(labels,
                                     tensor_projection,
                                     axis=-1)
        return features, projected_labels

    def project_l(labels):
        return tf.gather(labels, tensor_projection, axis=-1)

    if not(meta):
        return project_fl
    else:
        return project_l

def separate_image_from_label(*args):
    labels = args[:-1]
    path = args[-1]
    return path, tf.stack(labels, axis=0)

def drop_image(path, labels):
    return labels

def decode_img(img, image_shape): 
    img = tf.io.decode_image(tf.io.read_file(img), channels=3, dtype=tf.float32)
    img.set_shape(image_shape)
    return img

def load_image_from_path(image_shape):
    def fun(path, labels):
        return decode_img(path, image_shape), labels
    return fun

def intensity_to_activation(threshold=2):
    def fun(images, labels):
        return images, tf.dtypes.cast(labels >= threshold, dtype=tf.float32)
    return fun


def load_disfa_meta(fold, fold_repartition, image_size):
    dataset_file = FOLD_TEMPLATE.format(image_size, fold_repartition, fold)
    dataset_types = 12 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .cache())

def gen_disfa_v2(fold, batchsize, threshold=2, projection_mode="sota", fold_repartition='jaanet', image_size='299x299', meta=False, subsample=1):
    image_size_int = list(map(lambda x: int(x), image_size.split('x'))) + [3]
    meta_dataset = (load_disfa_meta(fold=fold,
                                    fold_repartition=fold_repartition,
                                    image_size=image_size)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .map(intensity_to_activation(threshold), tf.data.experimental.AUTOTUNE)
                    .shard(subsample, 0))

    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    projection = get_projection(projection_mode) 
    if meta:
        meta_dataset = (meta_dataset
                        .map(drop_image, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize)
                        .map(projective_map(projection, meta=meta), tf.data.experimental.AUTOTUNE)
                        .prefetch(tf.data.experimental.AUTOTUNE))
        return meta_dataset
    else:
        return (meta_dataset
                .shuffle(count)
                .map(load_image_from_path(image_size_int), tf.data.experimental.AUTOTUNE)
                .batch(batchsize)
                .map(projective_map(projection), tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE))

def gen_disfa_for_abaw(fold, batchsize, threshold=2, projection_mode="sota", fold_repartition='jaanet', image_size='299x299', meta=False, subsample=1):
    image_size_int = list(map(lambda x: int(x), image_size.split('x'))) + [3]
    meta_dataset = (load_disfa_meta(fold=fold,
                                    fold_repartition=fold_repartition,
                                    image_size=image_size)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .map(intensity_to_activation(threshold), tf.data.experimental.AUTOTUNE)
                    .shard(subsample, 0))

    count = meta_dataset.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1)
    if meta:
        meta_dataset = (meta_dataset
                        .map(drop_image, tf.data.experimental.AUTOTUNE)
                        .batch(batchsize)
                        .prefetch(tf.data.experimental.AUTOTUNE))
        return meta_dataset
    else:
        return (meta_dataset
                .shuffle(count)
                .map(load_image_from_path(image_size_int), tf.data.experimental.AUTOTUNE)
                .batch(batchsize)
                .prefetch(tf.data.experimental.AUTOTUNE))


"""
def gen_interleaved_disfa(fold, batchsize, mode, fold_repartition='jaanet', image_size='299x299', n_au=8, meta=False, subsample=1, intensity=False, interleave=True):
    image_size_int = list(map(lambda x: int(x), image_size.split('x'))) + [3]
    projection = SOTA_PROJECTION[:n_au] 
    if not(meta) and (mode=='train') and interleave:
        video_list = np.concatenate([fold_repartition_list(fold_repartition)[f] for f in fold])
        videos = [FRAMES_PER_VIDEO.format(video) for video in video_list]
        dataset_types = 12 * [float()] + [str()]
        video_datasets = [(tf.data.experimental.CsvDataset(video,
                                                           record_defaults=dataset_types,
                                                           header=True)
                           .cache()
                           .shuffle(4845)) for video in videos]
        counts = list(map(lambda x: x.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1),
                          video_datasets))
        max_counts = tf.math.reduce_max(counts)
        choice_dataset = (tf.data.Dataset.from_tensor_slices(tf.range(0, len(videos), dtype=tf.int64))
                          .shuffle(len(videos))
                          .repeat(max_counts))


        dataset = (tf.data.Dataset.choose_from_datasets(datasets=video_datasets,
                                                        choice_dataset=choice_dataset,
                                                        stop_on_empty_dataset=False)
                   .map(load_image(image_size_int), tf.data.experimental.AUTOTUNE)
                   .batch(batchsize)
                   .map(intensity_to_activation, tf.data.experimental.AUTOTUNE)
                   .map(projective_map(projection), tf.data.experimental.AUTOTUNE))
        return dataset
    else:
        return gen_disfa_v2(fold=fold,
                            batchsize=batchsize,
                            mode=mode,
                            fold_repartition=fold_repartition,
                            image_size=image_size,
                            n_au=n_au,
                            meta=meta, 
                            subsample=subsample)
"""


# DYSFER
def load_disfa_for_dysfer_meta(fold):
    dataset_file = os.path.join(DISFA, 'preprocessed', '299x299', 'jaanet_folds', '{}_dysfer.csv').format(fold)
    dataset_types = 5 * [float()] + [str()]
    return (tf.data.experimental.CsvDataset(dataset_file,
                                            record_defaults=dataset_types,
                                            header=True)
            .cache())

def gen_disfa_for_dysfer(batchsize, fold, threshold=2, subsample=1, meta=False, mix12=False):
    meta_dataset = (load_disfa_for_dysfer_meta(fold)
                    .map(separate_image_from_label, tf.data.experimental.AUTOTUNE)
                    .map(intensity_to_activation(threshold), tf.data.experimental.AUTOTUNE)
                    .shard(subsample, 0))
    if mix12:
        projection = [4, 2, 3]
    else:
        projection = [0, 1, 2, 3, 4]

    count = meta_dataset.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1)
    if meta:
        return (meta_dataset
                .map(drop_image, tf.data.experimental.AUTOTUNE)
                .batch(batchsize)
                .map(projective_map(projection, meta=meta), tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE))
    
    return (meta_dataset
            .shuffle(tf.dtypes.cast(count, tf.int64))
            .map(load_image_from_path((299, 299, 3)), tf.data.experimental.AUTOTUNE)
            .batch(batchsize)
            .map(projective_map(projection, meta=meta), tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE))



if __name__ == '__main__':
    disfa = gen_disfa([2], 8, 32, 'train', 'jaanet')
