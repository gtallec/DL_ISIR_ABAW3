import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm
import tensorflow as tf

from models.layers.normalisation_layers import InceptionNormalisation
from models.layers.dense import ParallelDense


class ToyAttentionEncoder(tkm.Model):
    def __init__(self, k, **kwargs):
        super(ToyAttentionEncoder, self).__init__(**kwargs)
        self.normalisation_layer = InceptionNormalisation()
        self.conv1 = tkl.Conv2D(filters=k,
                                kernel_size=32,
                                strides=32,
                                activation='linear',
                                padding='valid')
        self.k = k

    def call(self, x, training=None, **kwargs):
        batchsize = tf.shape(x)[0]
        x = self.normalisation_layer(x, training=training)
        x = self.conv1(x, training=training)
        print(x.shape)
        return tf.reshape(x, (batchsize, 25, self.k)) 


class ToyEncoder(tkm.Model):
    def __init__(self, bottleneck=128, **kwargs):
        super(ToyEncoder, self).__init__()

        self.dense1 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn1 = tkl.BatchNormalization()

        self.dense2 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn2 = tkl.BatchNormalization()

        self.dense3 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn3 = tkl.BatchNormalization()

        self.dense4 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn4 = tkl.BatchNormalization()

    def call(self, x, training=None, **kwargs):
        # (B, U)
        x_enc = self.dense1(x)
        x_enc = self.bn1(x_enc, training=training)

        # (B, U)
        x_enc = self.dense2(x_enc)
        x_enc = self.bn2(x_enc, training=training)

        x_enc = self.dense3(x_enc)
        x_enc = self.bn3(x_enc, training=training)

        x_enc = self.dense4(x_enc)
        x_enc = self.bn4(x_enc, training=training)
        return x_enc

class ToyAttentionOn1D(tkm.Model):
    def __init__(self, bottleneck=128, n_branch=4, **kwargs):
        super(ToyAttentionOn1D, self).__init__()

        self.dense1 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn1 = tkl.BatchNormalization()

        self.dense2 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn2 = tkl.BatchNormalization()

        self.dense3 = tkl.Dense(units=bottleneck, activation='relu')
        self.bn3 = tkl.BatchNormalization()

        self.dense4 = ParallelDense(units=bottleneck,
                                    activation='relu',
                                    T=n_branch)
        self.bn4 = tkl.BatchNormalization(axis=[1, 2])

        self.n_branch = n_branch

    def call(self, x, training=None, **kwargs):
        # (B, U)
        x_enc = self.dense1(x)
        x_enc = self.bn1(x_enc, training=training)

        # (B, U)
        x_enc = self.dense2(x_enc)
        x_enc = self.bn2(x_enc, training=training)

        # (B, U)
        x_enc = self.dense3(x_enc)
        x_enc = self.bn3(x_enc, training=training)

        # (B, n_branch, U)
        x_enc = tf.tile(tf.expand_dims(x_enc, axis=1),
                        multiples=(1, self.n_branch, 1))

        # (B, n_branch, U)
        x_enc = self.dense4(x_enc)
        x_enc = self.bn4(x_enc, training=training)
        return x_enc

if __name__ == '__main__':
    toy_encoder = ToyAttentionEncoder(k=64)
    toy_encoder.build((None, 160, 160, 3))
    toy_encoder.summary()
