import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm
import tensorflow as tf

from models.layers.attention_modules import MultiHeadAttention, MLPBlock


class PatchEncoder(tkm.Model):
    def __init__(self, num_patches, patch_size, d_model, positional_encoding=True, name='patch_encoder'):
        super(PatchEncoder, self).__init__(name=name)
        self.embedding_conv2d = tkl.Conv2D(filters=d_model,
                                           kernel_size=patch_size,
                                           strides=patch_size,
                                           padding='VALID')

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
            self.position_dense = tkl.Dense(units=d_model,
                                            activation='linear',
                                            kernel_initializer=embedding_init)
        self.num_patches = num_patches
        self.d_model = d_model
    
    def call(self, x):
        """ 
        Batch of images of size (B, H, W, C)
        """
        # (B, sqrt(num_patches), sqrt(num_patches), d_model)
        B = tf.shape(x)[0]
        encoded_image = self.embedding_conv2d(x)
        encoded_image = tf.reshape(encoded_image, (B, self.num_patches, self.d_model))

        if self.positional_encoding:
            # (1, N_p, N_p)
            positions = tf.expand_dims(tf.eye(self.num_patches), axis=0)
            # (1, N_p, d_model)
            positions = self.position_dense(positions)
            # (B, N_p, d_model)
            encoded_image = encoded_image + positions

        return encoded_image

class TimeDistributedPatchEncoder(tkm.Model):
    def __init__(self, num_patches, patch_size, d_model, positional_encoding=True, name='patch_encoder'):
        super(TimeDistributedPatchEncoder, self).__init__(name=name)
        self.embedding_conv2d = tkl.TimeDistributed(tkl.Conv2D(filters=d_model,
                                                               kernel_size=patch_size,
                                                               strides=patch_size,
                                                               padding='VALID'))

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
            self.position_dense = tkl.Dense(units=d_model,
                                            activation='linear',
                                            kernel_initializer=embedding_init)
        self.num_patches = num_patches
        self.d_model = d_model
    
    def call(self, x):
        """ 
        Batch of images of size (B, S, H, W, C)
        """
        # (B, sqrt(num_patches), sqrt(num_patches), d_model)
        B = tf.shape(x)[0]
        encoded_image = self.embedding_conv2d(x)
        encoded_image = tf.reshape(encoded_image, (B, -1, self.num_patches, self.d_model))

        if self.positional_encoding:
            # (N_p, N_p)
            positions = tf.eye(self.num_patches)
            # (N_p, d_model)
            positions = self.position_dense(positions)
            # (1, 1, N_p, d_model)
            positions = tf.reshape(positions, (1, 1, self.num_patches, self.d_model))
            # (B, S, N_p, d_model)
            encoded_image = encoded_image + positions

        return encoded_image
