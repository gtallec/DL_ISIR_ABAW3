import tensorflow.keras.models as tkm
import tensorflow as tf
class VGG_16(tkm.Model):
    def __init__(self, weights=None, **kwargs):
        super(VGG_16, self).__init__(**kwargs)
        self.vgg = tf.keras.applications.VGG16(include_top=False,
                                               weights=weights,
                                               input_tensor=None,
                                               input_shape=(224, 224, 3),
                                               pooling='avg')

    def call(self, inputs, training=None, **kwargs):
        return self.vgg(inputs, training=training)


SUPPORTED_VGG = {"vgg16": VGG_16}






