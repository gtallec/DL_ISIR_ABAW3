import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np

@tf.function
def random_flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


@tf.function
def random_rotation(image, deg):
    deg_rad = deg * np.pi / 180
    random_deg = tf.random.uniform([], minval=-deg_rad, maxval=deg_rad)
    return tfa.image.rotate(image, angles=random_deg)


@tf.function
def random_crop(image, pad):
    img_size = tf.shape(image)[0]
    image = tf.image.resize_with_crop_or_pad(image, img_size + pad, img_size + pad) 
    # Random crop back to the original size
    image = tf.image.random_crop(image, size=[img_size, img_size, 3])
    return image

def data_augmentation():
    def augment(image, label):
        # Random crop back to the original size
        image = random_rotation(image, 10) 
        # image = tf.image.random_hue(image, 0.2)
        image = random_flip(image)
        image = random_crop(image, 6)
        image = tf.clip_by_value(image, 0, 1)
        return image, label
    return augment
