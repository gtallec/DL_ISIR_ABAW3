import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow_addons as tfa
import tensorflow.keras.models as tkm

from models.layers.vit_layers import TimeDistributedPatchEncoder, PatchEncoder
from models.layers.sa_layers import SALayers 

class VisionTransformer(tkm.Model):
    def __init__(self,
                 patch_size,
                 num_patches,
                 d_model,
                 mlp_scale,
                 num_layers,
                 num_heads,
                 rate=0.1,
                 temp_xx=1.0,
                 name='vision_transformer',
                 **kwargs):

        super(VisionTransformer, self).__init__(name=name)
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads,
        self.rate = rate

        dff = d_model * mlp_scale

        self.patch_encoder = PatchEncoder(d_model=d_model,
                                          patch_size=patch_size,
                                          num_patches=num_patches)

        self.sa_layers = SALayers(d_model=d_model,
                                  num_heads=num_heads,
                                  num_layers=num_layers,
                                  dff=dff,
                                  rate=rate,
                                  temp=temp_xx,
                                  has_controller=False)
        
    def call(self, x, training=None, **kwargs):
        # (B, h_patch x v_patch, patch_dim)
        x = self.patch_encoder(x)
        # (B, N_p, d_model)
        x, blocks = self.sa_layers(x=x,
                                   mask=None,
                                   training=training)
        return x, blocks

class VideoTransformer(tkm.Model):
    def __init__(self,
                 patch_size,
                 num_patches,
                 d_model,
                 mlp_scale,
                 num_layers_space,
                 num_layers_time,
                 num_heads,
                 S,
                 rate=0.1,
                 temp_space=1.0,
                 temp_time=1.0,
                 name='video_transformer',
                 **kwargs):
        """
        max_len is the max_length of the timesequence given in input
        """
        super(VideoTransformer, self).__init__(name=name)
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_layers_space = num_layers_space
        self.num_layers_time = num_layers_time
        self.num_heads = num_heads
        self.rate = rate
        self.num_patches = num_patches 
        self.S = S

        dff = d_model * mlp_scale
        self.patch_encoder = TimeDistributedPatchEncoder(d_model=d_model,
                                                         patch_size=patch_size,
                                                         num_patches=num_patches)

        self.sa_layers_space = SALayers(d_model=d_model,
                                        num_heads=num_heads,
                                        num_layers=num_layers_space,
                                        dff=dff,
                                        rate=rate,
                                        temp=temp_space)

        self.sa_layers_time = SALayers(d_model=d_model,
                                       num_heads=num_heads,
                                       num_layers=num_layers_time,
                                       dff=dff,
                                       rate=rate,
                                       temp=temp_time)

        # Time Encoding
        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
        self.time_dense = tkl.Dense(units=d_model,
                                    activation='linear',
                                    kernel_initializer=embedding_init)


    def call(self, x, padding_mask, training=None, **kwargs):
        """
        x of size (B, S, N_p, d_model)
        padding_mask of size (B, S)
        """
        # Parallel patch encoding
        # (B, S, N_p, d_model)
        x = self.patch_encoder(x, training=training)
        B = tf.shape(x)[0]

        # time_attention_mask = tf.reshape(padding_mask, 
        # (S, S)
        time_positions = tf.eye(self.S)
        
        # Parallel self attention on images
        # (B, S, N_p, d_model)
        x, blocks = self.sa_layers(x=x,
                                   mask=None,
                                   training=training)

        # (1, S, 1, d_model)
        time_encodings = tf.reshape(self.time_dense(time_positions), 
                                    (1, self.S, 1, self.d_model))
        x = x + time_encodings
        # (B, S, 
        return x

if __name__ == '__main__':
    patch_size = 1  # Size of the patches to be extract from the input images
    num_patches = 9
    d_model = 64
    mlp_scale = 2
    num_heads = 4
    num_layers = 1
    rate = 0.1

    video_transformer = VideoTransformer(patch_size=patch_size,
                                         num_patches=num_patches,
                                         d_model=d_model,
                                         mlp_scale=mlp_scale,
                                         num_layers=num_layers,
                                         num_heads=num_heads,
                                         max_length=5,
                                         rate=rate,
                                         temp_xx=1.0)

    video_transformer.build((None, None, 3, 3, 1280))
    output = video_transformer(x=(tf.zeros((1, 1, 160, 160, 3))))
    tf.print(output.shape)
