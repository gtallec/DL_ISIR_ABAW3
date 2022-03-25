from models.encoders.vision_transformer import VisionTransformer
from models.encoders.inception import Inceptionv3

import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import tensorflow as tf

class InceptionViT(tkm.Model):
    def __init__(self,
                 d_model,
                 mlp_scale,
                 num_layers,
                 num_heads,
                 rate=0.1,
                 temp_xx=1.0,
                 weights=None,
                 **kwargs):
        super(InceptionViT, self).__init__()
        self.inception_v3 = Inceptionv3(pooling=None,
                                        weights=weights)
        self.vit = VisionTransformer(patch_size=1,
                                     num_patches=64,
                                     d_model=d_model,
                                     mlp_scale=mlp_scale,
                                     num_layers=num_layers,
                                     num_heads=num_heads,
                                     rate=rate,
                                     temp_xx=temp_xx)

    def call(self, x, training=None, **kwargs):
        # (B, N_p, N_p, dim_iv3)
        x = self.inception_v3(x, training=training)
        x, att_xx = self.vit(x, training=training)
        return x
