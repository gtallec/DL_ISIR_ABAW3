from models.layers.attention_modules_experimental import MultiHeadAttention, MLPBlock

import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

import tensorflow as tf

class SALayer(tkm.Model):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 temp=1.0,
                 rate=0.1):
        super(SALayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model,
                                      num_heads=num_heads,
                                      temp=temp)
        self.mlp_block = MLPBlock(d_model=d_model,
                                  dff=dff,
                                  rate=rate)

        self.layernorm1 = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tkl.Dropout(rate)
        self.dropout2 = tkl.Dropout(rate)

    def call(self, x, mask=None, training=None):
        """
        x (B, q, d_model): input queries
        mask (B, q, q): matrix of query authorizations for each element
        """
        mha_x, att_xx = self.mha(qkv=(x, x, x),
                                 mask=mask)
        mha_x = self.dropout1(mha_x, training=training)
        x = self.layernorm1(x + mha_x)

        mlp_x = self.mlp_block(x, training=training)
        mlp_x = self.dropout2(mlp_x, training=training)
        x = self.layernorm2(x + mlp_x)

        return x, att_xx
        
class SALayers(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 temp=1.0,
                 rate=0.1,
                 has_controller=False):
        super(SALayers, self).__init__()

        self.num_layers = num_layers
        self.sa_layers = [SALayer(d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  temp=temp,
                                  rate=rate)
                          for _ in range(num_layers)]

        self.has_controller = has_controller
        if has_controller:
            self.controller = tkl.Dense(units=d_model,
                                        kernel_initializer='zeros')

    def call(self, x, mask=None, training=None, **kwargs):
        """
        x (B, q, d_model): input queries
        mask (B, q, q): matrix of query authorizations for each element
        """
        atts_xx = []
        for i in range(self.num_layers):
            x, att_xx = self.sa_layers[i](x=x,
                                          mask=mask,
                                          training=training)
            atts_xx.append(att_xx)

        if self.has_controller:
            x = self.controller(x)
        return x, atts_xx

    def build(self, input_shape):
        print(input_shape)
        super().build(input_shape)
