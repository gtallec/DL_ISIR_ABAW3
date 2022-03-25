import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as tkl

# CUSTOM TRANSFORMATIONS
class MaskedDataAugmentation(tkl.Layer):
    def __init__(self, **kwargs):
        super(MaskedDataAugmentation, self).__init__()
    
    def call(self, x, y, training=None):
        """ 
        x (..., H, W, C)
        y : (..., T)
        """
        if training:
            # Compute padding mask
            batch_shape = tf.shape(x)[:-3]
            x_shape = tf.shape(x)[-3:]
            y_shape = tf.shape(y)[-1:]

            batch_rank = tf.shape(batch_shape)[0]
            total_rank = batch_rank + tf.shape(x_shape)[0]
            x_axis = tf.range(start=batch_rank, limit=total_rank)
            # (..., )
            padding_mask = 1 - tf.dtypes.cast(tf.math.reduce_sum(tf.math.abs(x), axis=x_axis) == 0, dtype=tf.float32)

            padding_mask_x = tf.reshape(padding_mask, tf.concat([tf.shape(padding_mask), tf.ones_like(x_shape)], axis=0))
            padding_mask_y = tf.reshape(padding_mask, tf.concat([tf.shape(padding_mask), tf.ones_like(y_shape)], axis=0)) 
            # (B, S, H, W, C), (B, S, T) 
            aug_x, aug_y = self.augmentation(x, y, padding_mask=padding_mask)
            aug_x, aug_y = padding_mask_x * aug_x, padding_mask_y * aug_y
            # Padded frames remain padded after augmentation
            return aug_x, aug_y
        else:
            return x, y

    def augmentation(self, x, y):
        return x, y

class MaskedMixUp(MaskedDataAugmentation):
    """
    Implement mixup i.e intra-batch random linear combination of images and labels.
    """
    def __init__(self, beta_mixup, ignore_mixup=False, **kwargs):
        super(MaskedMixUp, self).__init__()
        self.beta_mixup = beta_mixup
        self.ignore_mixup = ignore_mixup
        self.beta_distribution = tfp.distributions.Beta(concentration1=self.beta_mixup,
                                                        concentration0=self.beta_mixup)

    def augmentation(self, x, y, padding_mask, **kwargs):
        """ 
        x (..., H, W, C)
        y : (..., T)
        padding mask : (..., )
        """
        if not(self.ignore_mixup):
            batchseq = tf.shape(padding_mask)
            batchflat = [tf.math.reduce_prod(batchseq)]

            shape_img = tf.shape(x)[-3:]
            x = tf.reshape(x, tf.concat([batchflat, shape_img], axis=0))

            T = tf.shape(y)[-1:]
            y = tf.reshape(y, tf.concat([batchflat, T], axis=0))
            # Sample convex combination
            # (..., )
            alphas = self.beta_distribution.sample(sample_shape=batchflat)
            alphas_x = tf.reshape(alphas, shape=tf.concat([batchflat, tf.ones_like(shape_img)], axis=0))
            alphas_y = tf.reshape(alphas, shape=tf.concat([batchflat, tf.ones_like(T)], axis=0))

            # Sample Paired images :
            # All frames can be selected uniformly:
            sampling_logits = tf.math.log((1. / tf.dtypes.cast(batchflat[0], tf.float32)
                                           * tf.ones(batchflat)))
            # Except the masked ones:
            padding_mask = tf.reshape(padding_mask, batchflat)
            # (..., )
            masked_sampling_logits = sampling_logits + (1 - padding_mask) * -1e9
            # (..., ...)
            masked_sampling_logits = tf.tile(masked_sampling_logits[tf.newaxis, :],
                                             tf.concat([batchflat, [1]], axis=0))

            # (..., )
            paired_indices = tf.squeeze(tf.random.categorical(logits=masked_sampling_logits,
                                                              num_samples=1),
                                        axis=-1)
            paired_x = tf.gather(x, paired_indices, axis=0)
            paired_y = tf.gather(y, paired_indices, axis=0)

            mixed_x = alphas_x * x + (1.0 - alphas_x) * paired_x
            mixed_y = alphas_y * y + (1.0 - alphas_y) * paired_y
            
            mixed_x = tf.reshape(mixed_x, tf.concat([batchseq, shape_img], axis=0))
            mixed_y = tf.reshape(mixed_y, tf.concat([batchseq, T], axis=0))
            return mixed_x, mixed_y
        return x, y

class MaskedLabelSmoothing(MaskedDataAugmentation):
    """
    Implement label smoothing i.e replace one hot labels y_oh by (1 - alpha) y_oh + alpha/K.
    """
    def __init__(self, alpha_smooth, K, **kwargs):
        super(MaskedLabelSmoothing, self).__init__()
        self.alpha_smooth = alpha_smooth
        self.K = K

    def augmentation(self, x, y, **kwargs):
        return x, (1.0 - self.alpha_smooth) * y + self.alpha_smooth / self.K

# KERAS LAYER BASED TRANSFORMATION
class MaskedInputProcessing(tkl.Layer):
    def __init__(self, processing_layer, **kwargs):
        super(MaskedInputProcessing, self).__init__()
        self.processing_layer = processing_layer

    def call(self, x, y, training=None):
        """ x of size (B, S, H, W, C) """
        # Compute padding mask
        batch_shape = tf.shape(x)[:-3]
        batchflat = tf.math.reduce_prod(batch_shape)
        x_shape = tf.shape(x)[-3:]

        batch_rank = tf.shape(batch_shape)[0]
        total_rank = batch_rank + tf.shape(x_shape)[0]
        x_axis = tf.range(start=batch_rank, limit=total_rank)
        # (..., )
        padding_mask = 1 - tf.dtypes.cast(tf.math.reduce_sum(tf.math.abs(x), axis=x_axis) == 0, dtype=tf.float32)
        padding_mask = tf.reshape(padding_mask, tf.concat([tf.shape(padding_mask), tf.ones_like(x_shape)], axis=0))

        # (B x S, H, W, C)
        x = tf.reshape(x, tf.concat([[batchflat], x_shape], axis=0))
        aug_x = self.processing_layer(x, training=training)
        aug_x_shape = tf.shape(aug_x)[-3:]
        aug_x = tf.reshape(aug_x, tf.concat([batch_shape, aug_x_shape], axis=0))

        return padding_mask * aug_x, y

# GEOMETRIC TRANSFORMATION
class MaskedResize(MaskedInputProcessing):
    def __init__(self, im_h, im_w, **kwargs):
        super(MaskedResize, self).__init__(tkl.Resizing(height=im_h,
                                                        width=im_w))

class MaskedRandomHorizontalFlip(MaskedInputProcessing):
    def __init__(self, **kwargs):
        super(MaskedRandomHorizontalFlip, self).__init__(tkl.RandomFlip(mode='horizontal'))

class MaskedRandomRotation(MaskedInputProcessing):
    def __init__(self, rotation_factor, **kwargs):
        super(MaskedRandomRotation, self).__init__(tkl.RandomRotation(factor=rotation_factor))

class MaskedRandomZoom(MaskedInputProcessing):
    def __init__(self, zoom_factor, **kwargs):
        super(MaskedRandomZoom, self).__init__(tkl.RandomZoom(zoom_factor))

# ILLUMINATION TRANSFORMATION
class MaskedRandomContrast(MaskedInputProcessing):
    def __init__(self, contrast_factor, **kwargs):
        super(MaskedRandomContrast, self).__init__(tkl.RandomContrast(factor=contrast_factor))

class MaskedRandomBrightness(MaskedDataAugmentation):
    def __init__(self, brightness_factor, **kwargs):
        super(MaskedRandomBrightness, self).__init__()
        self.brightness_factor = brightness_factor

    def augmentation(self, x, y, **kwargs):
        """
        x of size (..., H, W, C)
        y of size (..., T)
        """
        batchseq = tf.shape(x)[:-3]
        img_shape_ones = tf.ones_like(tf.shape(x)[-3:])

        # (B, 1, 1, 1)
        brightness_noise = tf.random.uniform(shape=tf.concat([batchseq, img_shape_ones], axis=0),
                                             minval=-self.brightness_factor,
                                             maxval=self.brightness_factor)
        return tf.clip_by_value(x + brightness_noise,
                                clip_value_min=0.0,
                                clip_value_max=1.0), y

# Color Transformation
class MaskedChannelDrop(MaskedDataAugmentation):
    """
    Implement Channel Drop which independently drop channels with proba p_channel 
    """
    def __init__(self, p_channel, **kwargs):
        super(MaskedChannelDrop, self).__init__()
        self.p_channel = p_channel

    def augmentation(self, x, y, **kwargs):
        """
        x of size (..., H, W, C)
        y of size (..., T)
        """
        batchseq = tf.shape(x)[:-3]
        im_size = tf.shape(x)[-3:-1]
        C = tf.shape(x)[-1:]
        mask_shape = tf.concat([batchseq, tf.ones_like(im_size), C], axis=0)
        # (..., 1, 1, C)
        mask = tf.dtypes.cast(tf.random.uniform(shape=mask_shape,
                                                minval=0,
                                                maxval=1)
                              - self.p_channel >= 0, dtype=tf.float32)
        return x * mask, y


SUPPORTED_MASKED_DATA_PROCESSING_LAYERS = {"m_resize": MaskedResize,
                                           "m_horizontal_flip": MaskedRandomHorizontalFlip,
                                           "m_rotation": MaskedRandomRotation,
                                           "m_zoom": MaskedRandomZoom,
                                           "m_contrast": MaskedRandomContrast,
                                           "m_channel_drop": MaskedChannelDrop,
                                           "m_brightness": MaskedRandomBrightness,
                                           "m_mixup": MaskedMixUp,
                                           "m_label_smoothing": MaskedLabelSmoothing}
