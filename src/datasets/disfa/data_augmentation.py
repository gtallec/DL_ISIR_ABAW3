import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

import numpy as np

def random_flip(images):
    """
    In : images of size (B, H, W, C)
    Out : randomly flipped images of size (B, H, W, C)
    """
    images = tf.image.random_flip_left_right(images)
    return images

def random_rotation(images, deg):
    """
    In : images of size (B, H, W, C)
    Out : randomly flipped images of size (B, H, W, C)
    """
    batchsize = images.shape[0]
    deg_rad = deg * np.pi / 180
    random_degs = tf.random.uniform(shape=(batchsize, ), minval=-deg_rad, maxval=deg_rad)
    return tfa.image.rotate(images, angles=random_degs)

def random_crop(image, pad):
    img_size = tf.shape(image)[0]
    image = tf.image.resize_with_crop_or_pad(image, img_size + pad, img_size + pad) 
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[img_size, img_size, 3])
    return image

def mixup(images, labels, beta=0.4):
    """ 
    images of size (B, H, W, C)
    labels of size (B, T)
    """
    shape = tf.shape(images)
    batchsize, H, W, C = shape[0], shape[1], shape[2], shape[3]
    T = tf.shape(labels)[1]
    beta_distributions = tfp.distributions.Beta(concentration1=beta,
                                                concentration0=beta)
    # Sample convex combination
    # (B, )
    alphas = beta_distributions.sample(sample_shape=(batchsize, ))
    alphas_images = tf.tile(alpha[:, tf.newaxis, tf.newaxis, tf.newaxis],
                            (1, H, W, C))
    alphas_labels = tf.tile(alpha[:, tf.newaxis],
                            (1, T))

    # Sample Paired images :
    paired_indices = tf.random.uniform(shape=(batchsize, ), minval=0, maxval=batchsize, dtype=tf.int32)
    paired_images = tf.gather(images, paired_indices, axis=0)
    paired_labels = tf.gather(labels, paired_indices, axis=0)

    return (alphas_images * images + (1 - alphas_images) * paired_images,
            alphas_labels * labels + (1 - alphas_labels) * paired_labels)

def data_augmentation():
    def augment(images, labels):
        # Random crop back to the original size
        images = random_rotation(images, 10) 
        # image = tf.image.random_hue(image, 0.2)
        images = random_flip(images)
        # image = random_crop(image, 6)
        images = tf.clip_by_value(images, 0, 1)
        return images, labels
    return augment
