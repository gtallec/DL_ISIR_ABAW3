import tensorflow.keras.layers as tkl
import tensorflow as tf

from models.layers.attention_modules import MultiHeadAttention, MLPBlock
from models.layers.ca_layers import CALayer
from models.layers.sa_layers import SALayer


class TTALayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 temp_tt,
                 temp_tx,
                 rate=0.1):
        super(TTALayer, self).__init__()
        # Token Self Attention
        self.sa_tt = SALayer(d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             temp=temp_tt,
                             rate=rate)

        self.ca_tx = CALayer(d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             temp=temp_tx,
                             rate=rate)

    def call(self, tokens, x, mask=None, training=None):
        tokens, att_tt = self.sa_tt(x=tokens,
                                    mask=mask,
                                    training=training)
        tokens, att_tx = self.ca_tx(query=tokens,
                                    key=x,
                                    mask=mask,
                                    training=training)
        return tokens, att_tt, att_tx


class STATLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 temp_tt,
                 temp_tx,
                 rate=0.1):
        super(STATLayer, self).__init__()
        self.d_model = d_model
        self.T = T
        # Token Self Attention
        self.sa_tt = SALayer(d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             temp=temp_tt,
                             rate=rate)

        self.ca_tx = CALayer(d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             temp=temp_tx,
                             rate=rate)

    def call(self, tokens, x, mask=None, training=None):
        """
        tokens (B, S, T, d_model)
        x (B, S, N, d_model)
        """
        # (B, S, T, d_model)
        B = tf.shape(tokens)[0]
        # (B, S * T, d_model)
        tokens = tf.reshape(tokens, (B, -1, self.d_model))
        # (B, S * T, d_model)
        tokens, att_tt = self.sa_tt(x=tokens,
                                    training=training)

        d_model_x = tf.shape(x)[-1]
        # (B, S * T, d_model_x)
        x = tf.reshape(x, (B, -1, d_model_x))
        # (B, S * T, d_model)
        tokens, att_tx = self.ca_tx(query=tokens,
                                    key=x,
                                    training=training)
        tokens = tf.reshape(tokens, (B, -1, self.T, self.d_model))
        return tokens, att_tt, att_tx

class TTALayers(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 temp_tt,
                 temp_tx,
                 rate=0.1):
        super(TTALayers, self).__init__()
        self.num_layers = num_layers
        self.tta_layers = [TTALayer(d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    temp_tt=temp_tt,
                                    temp_tx=temp_tx,
                                    rate=0.1)
                           for _ in range(self.num_layers)]

    def call(self, tokens, x, mask=None, training=None, **kwargs):
        atts_tt = []
        atts_tx = []
        for i in range(self.num_layers):
            tokens, att_tt, att_tx = self.tta_layers[i](tokens=tokens,
                                                        x=x,
                                                        mask=mask,
                                                        training=training)
            atts_tt.append(att_tt)
            atts_tx.append(att_tx)
        return tokens, atts_tt, atts_tx

class STATLayers(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 temp_tt,
                 temp_tx,
                 rate=0.1):
        super(STATLayers, self).__init__()
        self.num_layers = num_layers
        self.stat_layers = [STATLayer(d_model=d_model,
                                      num_heads=num_heads,
                                      dff=dff,
                                      temp_tt=temp_tt,
                                      temp_tx=temp_tx,
                                      T=T,
                                      rate=0.1)
                            for _ in range(self.num_layers)]

    def call(self, tokens, x, mask=None, training=None, **kwargs):
        atts_tt = []
        atts_tx = []
        for i in range(self.num_layers):
            tokens, att_tt, att_tx = self.stat_layers[i](tokens=tokens,
                                                         x=x,
                                                         mask=mask,
                                                         training=training)
            atts_tt.append(att_tt)
            atts_tx.append(att_tx)
        return tokens, atts_tt, atts_tx
