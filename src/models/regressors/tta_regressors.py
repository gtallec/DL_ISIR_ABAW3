import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

from models.layers.tta_layers import TTALayers

class TTAT(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 mlp_scale,
                 T,
                 temp_tt=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 **kwargs):
        super(TTAT, self).__init__()


        self.num_heads = num_heads
        self.d_model = d_model
        dff = self.d_model * mlp_scale


        self.num_layers = num_layers
        self.tta_layers = TTALayers(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    temp_tt=temp_tt,
                                    temp_tx=temp_tx,
                                    rate=rate)
        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.0)

        self.token_embedder = tkl.Dense(units=d_model,
                                        activation='linear',
                                        kernel_initializer=embedding_init,
                                        use_bias=True)

        self.final_dense = tkl.Dense(units=1,
                                     activation='linear')

        self.T = T

    def call(self, inputs, training=None, **kwargs):
        """ 
        inputs of shape (B, N_x, d_model)
        """
        output_dict = dict()
        B = tf.shape(inputs)[0]
        # (1, T, T) -> (B, T, T)
        tokens = tf.expand_dims(tf.eye(self.T), axis=0)
        tokens = tf.tile(tokens, (B, 1, 1))

        # (B, T, d)
        tokens = self.token_embedder(tokens)

        # (B, T, d), (B, H, T, N_x
        prelogits, atts_tt, atts_tx = self.tta_layers(tokens=tokens,
                                                      x=inputs,
                                                      mask=None,
                                                      training=training)
        # (B, T)
        logits = self.final_dense(prelogits)
        logits = tf.reshape(logits, (B, self.T))
        output_dict['loss'] = logits
        output_dict['global_pred'] = tf.math.sigmoid(logits)

        for i in range(self.num_layers):
            output_dict['layer_{}_att_tt'.format(i)] = atts_tt[i]
            output_dict['layer_{}_att_tx'.format(i)] = atts_tx[i]

        return output_dict
