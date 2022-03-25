import tensorflow as tf

import tensorflow.keras.layers as tkl


class InceptionNormalisation(tkl.Layer):
    def __init__(self, **kwargs):
        super(InceptionNormalisation, self).__init__(**kwargs)

    def call(self, x, training=None, **kwargs):
        return tf.image.per_image_standardization(2 * x - 1)

class EfficientNetNormalisation(tkl.Layer):
    def __init__(self, **kwargs):
        super(EfficientNetNormalisation, self).__init__(**kwargs)

    def call(self, x, training=None, **kwargs):
        return 255 * x
