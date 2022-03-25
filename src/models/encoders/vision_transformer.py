import tensorflow as tf
import tensorflow.keras.models as tkm

from models.layers.vit_layers import  PatchEncoder
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

