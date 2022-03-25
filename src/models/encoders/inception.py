import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

import tensorflow as tf

class Inceptionv3(tkm.Model):
    def __init__(self, pooling=None, weights=None, **kwargs):
        super(Inceptionv3, self).__init__()
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                           weights=weights,
                                                                           pooling=pooling)

    def call(self, inputs, training=None, **kwargs):
        x = tf.keras.applications.inception_v3.preprocess_input(255 * inputs)
        return self.inception_v3(x, training=training)

class Inceptionv3Attention(tkm.Model):
    def __init__(self, d_model, weights=None, **kwargs):
        super(Inceptionv3Attention, self).__init__()
        self.d_model = d_model
        self.inception_v3 = Inceptionv3(pooling=None,
                                        weights=weights)
        self.compresser = tkl.Dense(units=d_model)

    def call(self, inputs, training=None, **kwargs):
        B = tf.shape(inputs)[0]
        # (B, N_patch, N_patch, dim_f)
        x = self.inception_v3(inputs, training=training)
        x = tf.reshape(x, (B, 64, 2048))
        # (B, 64, d_model)
        x = self.compresser(x)
        return x
