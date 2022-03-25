import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import tensorflow as tf
class EfficientNetB0(tkm.Model):
    def __init__(self, weights=None, pooling=None, **kwargs):
        super(EfficientNetB0, self).__init__(**kwargs)
        self.efficientnetb0 = tf.keras.applications.EfficientNetB0(include_top=False,
                                                                   weights=weights,
                                                                   pooling=pooling)

    def call(self, inputs, training=None, **kwargs):
        output = self.efficientnetb0(inputs, training=training)
        return output 


class EfficientNetB7(tkm.Model):
    def __init__(self, pooling=None, weights=None, **kwargs):
        super(EfficientNetB7, self).__init__(**kwargs)
        self.efficientnetb7 = tf.keras.applications.EfficientNetB7(include_top=False,
                                                                   weights=weights,
                                                                   pooling=pooling)
        self.pooling = pooling

    def call(self, inputs, training=None, **kwargs):
        output = self.efficientnetb7(inputs, training=training)
        return output

class EfficientNetB0Layer(tkl.Layer):
    def __init__(self, pooling=None, weights=None, **kwargs):
        super(EfficientNetB0Layer, self).__init__()
        self.efficientnetb0 = tf.keras.applications.EfficientNetB0(include_top=False,
                                                                   weights=weights,
                                                                   pooling=pooling)

    def call(self, inputs, training=None, **kwargs):
        inputs = inputs * 255
        return self.efficientnetb0(inputs, training=training)

    def compute_output_shape(self, input_shape):
        self.efficientnetb0.build(input_shape)
        output_shape = self.efficientnetb0(tf.zeros((1, *input_shape[1:]))).shape
        final_shape = (None, *output_shape[1:])
        return final_shape

class TimeDistributedEfficientNetB0ForAttention(tkm.Model):
    def __init__(self, pooling=None, weights=None, **kwargs):
        super(TimeDistributedEfficientNetB0ForAttention, self).__init__(**kwargs) 
        self.td_efficientnetb0 = tkl.TimeDistributed(EfficientNetB0Layer(weights=weights,
                                                                         pooling=pooling))

    def call(self, inputs, training=None, **kwargs):
        """ inputs (B, S, H, W, C) is a batch of S images """
        return self.td_efficientnetb0(inputs, training=training)

class TimeDistributedEfficientNetB0(tkm.Model):
    def __init__(self, S, pooling, weights, bottleneck, **kwargs):
        super(TimeDistributedEfficientNetB0, self).__init__()
        self.td_efficientnetb0 = TimeDistributedEfficientNetB0ForAttention(pooling=pooling,
                                                                           weights=weights)
        self.dense = tkl.Dense(units=bottleneck, activation='linear')
        self.S = S

    def call(self, inputs, training=None, **kwargs):
        """ inputs (B, S, H, W, C) is a batch of sequences of S images """
        # Build padding mask if images are full of zeros
        # (B, S, H, W, C)
        input_shape = tf.shape(inputs)
        B = input_shape[0]
        # (B, S, H, W, C)
        efficient_encoding = self.td_efficientnetb0(inputs, training=training)
        # (B, S x C)
        C = tf.shape(efficient_encoding)[-1]
        efficient_encoding = tf.reshape(efficient_encoding, (B, self.S*C))
        # (B, H)
        return self.dense(efficient_encoding)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    efficientnetb0 = TimeDistributedEfficientNetB0(pooling='avg', S=2, weights='imagenet', bottleneck=512)
    output_shape = efficientnetb0.compute_output_shape((None, None, 112, 112, 3))
    output = efficientnetb0(tf.ones((1, 2, 112, 112, 3)))
    tf.print(output.shape)
