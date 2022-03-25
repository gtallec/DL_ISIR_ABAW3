import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow.keras.layers as tkl

import numpy as np


# Normalization layers
def input_from_dataset(inputs, labels):
    return inputs

def sum_for_moments(reduce_axis):
    def fun(accu, el):
        counts, sums, squared_sums = accu
        count_new = tf.dtypes.cast(tf.math.reduce_prod(tf.gather(tf.shape(el), reduce_axis)), dtype=tf.float32)
        sums_new = tf.math.reduce_sum(el, axis=reduce_axis)
        squared_sums_new = tf.math.reduce_sum(tf.math.pow(el, 2), axis=reduce_axis) 
        return counts + count_new, sums + sums_new, squared_sums + squared_sums_new
    return fun

def mean_and_variance(dataset, input_shape, reduce_axis, keep_axis):
    mean_and_var_shape = tuple(input_shape[d] for d in keep_axis)
    count_init = tf.constant(0, dtype=tf.float32)
    sum_init = tf.zeros(mean_and_var_shape)
    squared_sum_init = tf.zeros(mean_and_var_shape)
    counts, sums, squared_sums = dataset.reduce((count_init, sum_init, squared_sum_init),
                                                sum_for_moments(reduce_axis))
    mean = sums / counts
    return mean, (squared_sums / counts) - tf.math.pow(mean, 2)


class Normalization(tkl.Layer):
    def __init__(self, dataset=None, axis=-1, **kwargs):
        super(Normalization, self).__init__()
        self.normalization = tkl.Normalization(axis=axis)
        if dataset is not None:
            input_dataset = dataset.map(input_from_dataset, tf.data.experimental.AUTOTUNE)
        self.normalization.adapt(input_dataset)

    def call(self, x, y=None, **kwargs):
        return self.normalization(x), y


class CenterReduceDeprecated(tkl.Layer):
    def __init__(self, train_dataset=None, axis=-1, **kwargs):
        super(CenterReduce, self).__init__()
        if axis is None:
            axis = ()
        elif isinstance(axis, int):
            axis = (axis,)
        else:
            axis = tuple(axis)
        self.axis = axis
        self.train_mode = train_dataset is not None

        if self.train_mode:
            self.dataset = train_dataset.map(input_from_dataset, tf.data.experimental.AUTOTUNE)
        else:
            self.dataset = train_dataset

    def build(self, input_shape):
        ndim = len(input_shape)
        self._keep_axis = sorted([d if d >= 0 else d + ndim for d in self.axis])
        self._reduce_axis = [d for d in range(ndim) if d not in self._keep_axis]
        mean_and_var_shape = tuple(input_shape[d] for d in self._keep_axis)
        if self.train_mode:
            mean, variance = mean_and_variance(dataset=self.dataset,
                                               input_shape=input_shape,
                                               reduce_axis=self._reduce_axis,
                                               keep_axis=self._keep_axis)
        else:
            mean, variance = tf.zeros(mean_and_var_shape), tf.ones(mean_and_var_shape)

        self.mean = tf.Variable(initial_value=mean,
                                trainable=False)
        self.variance = tf.Variable(initial_value=variance,
                                    trainable=False)

    def call(self, x, y=None, **kwargs):
        return ((x - self.mean) / tf.maximum(tf.sqrt(self.variance), tf.keras.backend.epsilon())), y

# CUSTOM TRANSFORMATIONS
class TrainDataAugmentation(tkl.Layer):
    def __init__(self, **kwargs):
        super(TrainDataAugmentation, self).__init__()
    
    def call(self, x, training=None, y=None):
        if training:
            return self.augmentation(x, y)
        else:
            return x, y

    def augmentation(self, x, y):
        return x, y

class MixUp(TrainDataAugmentation):
    """
    Implement mixup i.e intra-batch random linear combination of images and labels.
    """
    def __init__(self, beta_mixup, ignore_mixup=False, **kwargs):
        super(MixUp, self).__init__()
        self.beta_mixup = beta_mixup
        self.ignore_mixup = ignore_mixup
        self.beta_distribution = tfp.distributions.Beta(concentration1=self.beta_mixup,
                                                        concentration0=self.beta_mixup)

    def augmentation(self, x, y):
        if not(self.ignore_mixup):
            batchsize = tf.shape(x)[0]
            # Sample convex combination
            # (B, )
            alphas = self.beta_distribution.sample(sample_shape=(batchsize, ))
            alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
            alphas_y = alphas[:, tf.newaxis]

            # Sample Paired images :
            paired_indices = tf.random.uniform(shape=(batchsize, ), minval=0, maxval=batchsize, dtype=tf.int32)
            paired_x = tf.gather(x, paired_indices, axis=0)
            paired_y = tf.gather(y, paired_indices, axis=0)

            mixed_x = alphas_x * x + (1.0 - alphas_x) * paired_x
            mixed_y = alphas_y * y + (1.0 - alphas_y) * paired_y

            return mixed_x, mixed_y
        return x, y

class AUCutMix(TrainDataAugmentation):
    """
    Implement mixup i.e intra-batch random linear combination of images and labels.
    """
    def __init__(self, p_cutmix, h_cut1=60, h_cut2=80, **kwargs):
        super(AUCutMix, self).__init__()
        self.p_cutmix = p_cutmix
        self.h_cut1 = h_cut1
        self.h_cut2 = h_cut2

    def augmentation(self, x, y):
        shape_x = tf.shape(x)
        batchsize = shape_x[0]
        H, W, C = shape_x[1], shape_x[2], shape_x[3]
        
        up = tf.ones((self.h_cut1, W, C))
        mid = tf.ones((self.h_cut2 - self.h_cut1, W, C))
        low = tf.ones((H - self.h_cut2, W, C))

        uzone_x = tf.concat([up, 0 * mid, 0 * low], axis=0)[tf.newaxis, :, :, :]
        mzone_x = tf.concat([0 * up, mid, 0 * low], axis=0)[tf.newaxis, :, :, :]
        lzone_x = tf.concat([0 * up, 0 * mid, low], axis=0)[tf.newaxis, :, :, :]

        uzone_y = tf.constant([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)[tf.newaxis, :]
        mzone_y = tf.constant([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)[tf.newaxis, :]
        lzone_y = tf.constant([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=tf.float32)[tf.newaxis, :]
        # (B, )
        is_cut = tf.dtypes.cast(tf.random.uniform(shape=(batchsize,),
                                                  minval=0,
                                                  maxval=1)
                                -
                                self.p_cutmix <= 0, dtype=tf.float32)
        is_cut_x = is_cut[:, tf.newaxis, tf.newaxis, tf.newaxis]
        is_cut_y = is_cut[:, tf.newaxis]


        # Sample Paired images :
        paired_indices = tf.random.uniform(shape=(batchsize, ), minval=0, maxval=batchsize, dtype=tf.int32)
        
        upper_x = x
        lower_x = tf.gather(x, paired_indices, axis=0)

        upper_y = y
        lower_y = tf.gather(y, paired_indices, axis=0)

        mixed_x = (uzone_x * upper_x +
                   lzone_x * lower_x +
                   0.5 * mzone_x * (upper_x + lower_x))
        mixed_y = (uzone_y * upper_y +
                   lzone_y * lower_y +
                   0.5 * mzone_y * (upper_y + lower_y))
        
        output_x = is_cut_x * mixed_x + (1 - is_cut_x) * x
        output_y = is_cut_y * mixed_y + (1 - is_cut_y) * y

        return output_x, output_y

class LabelSmoothing(TrainDataAugmentation):
    """
    Implement label smoothing i.e replace one hot labels y_oh by (1 - alpha) y_oh + alpha/K.
    """
    def __init__(self, alpha_smooth, K, **kwargs):
        super(LabelSmoothing, self).__init__()
        self.alpha_smooth = alpha_smooth
        self.K = K

    def augmentation(self, x, y):
        return x, (1.0 - self.alpha_smooth) * y + self.alpha_smooth / self.K

class PartialLabelSmoothing(TrainDataAugmentation):
    """
    Implement label smoothing for partially known labelled.
    Only the known labels are smoothed.
    """
    def __init__(self, alpha_smooth, K, **kwargs):
        super(PartialLabelSmoothing, self).__init__()
        self.alpha_smooth = alpha_smooth
        self.K = K

    def augmentation(self, x, y):
        # (B, T)
        partial_labels_mask = 1 - tf.dtypes.cast(y == -1, dtype=tf.float32)
        # (B, T)
        smooth_y = (1.0 - self.alpha_smooth) * y + self.alpha_smooth / self.K
        aug_y = partial_labels_mask * smooth_y + (1 - partial_labels_mask) * y
        return x, aug_y

# KERAS LAYER BASED TRANSFORMATION
class InputProcessing(tkl.Layer):
    def __init__(self, processing_layer, **kwargs):
        super(InputProcessing, self).__init__()
        self.processing_layer = processing_layer

    def call(self, x, training=None, y=None):
        return self.processing_layer(x, training=training), y


# GEOMETRIC TRANSFORMATION
class Resize(InputProcessing):
    def __init__(self, im_h, im_w, **kwargs):
        super(Resize, self).__init__(tkl.Resizing(height=im_h,
                                                  width=im_w))

class RandomHorizontalFlip(InputProcessing):
    def __init__(self, **kwargs):
        super(RandomHorizontalFlip, self).__init__(tkl.RandomFlip(mode='horizontal'))

class RandomRotation(InputProcessing):
    def __init__(self, rotation_factor, **kwargs):
        super(RandomRotation, self).__init__(tkl.RandomRotation(factor=rotation_factor))

class RandomZoom(InputProcessing):
    def __init__(self, zoom_factor, **kwargs):
        super(RandomZoom, self).__init__(tkl.RandomZoom(zoom_factor))

# ILLUMINATION TRANSFORMATION
class RandomContrast(InputProcessing):
    def __init__(self, contrast_factor, **kwargs):
        super(RandomContrast, self).__init__(tkl.RandomContrast(factor=contrast_factor))

class RandomBrightness(TrainDataAugmentation):
    def __init__(self, brightness_factor, **kwargs):
        super(RandomBrightness, self).__init__()
        self.brightness_factor = brightness_factor

    def augmentation(self, x, y):
        batchsize = tf.shape(x)[0]
        # (B, 1, 1, 1)
        brightness_noise = tf.random.uniform(shape=(batchsize, ),
                                             minval=-self.brightness_factor,
                                             maxval=self.brightness_factor)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        return tf.clip_by_value(x + brightness_noise,
                                clip_value_min=0.0,
                                clip_value_max=1.0), y

# Color Transformation
class ChannelDrop(TrainDataAugmentation):
    """
    Implement Channel Drop which independently drop channels with proba p_channel 
    """
    def __init__(self, p_channel, **kwargs):
        super(ChannelDrop, self).__init__()
        self.p_channel = p_channel

    def augmentation(self, x, y):
        batchsize = tf.shape(x)[0]
        # (B, 3)
        mask = 1 - tf.dtypes.cast(tf.random.uniform(shape=(batchsize, 3),
                                                    minval=0,
                                                    maxval=1)
                                  - self.p_channel <= 0, dtype=tf.float32)
        return x * mask[:, tf.newaxis, tf.newaxis, :], y




SUPPORTED_BASIC_DATA_PROCESSING_LAYERS = {"resize": Resize,
                                          "horizontal_flip": RandomHorizontalFlip,
                                          "rotation": RandomRotation,
                                          "zoom": RandomZoom,
                                          "contrast": RandomContrast,
                                          "channel_drop": ChannelDrop,
                                          "brightness": RandomBrightness,
                                          "mixup": MixUp,
                                          "au_cutmix": AUCutMix,
                                          "label_smoothing": LabelSmoothing,
                                          "pl_label_smoothing": PartialLabelSmoothing,
                                          "normalization": Normalization}

