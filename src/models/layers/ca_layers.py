from models.layers.attention_modules import MultiHeadAttention, MLPBlock

import tensorflow.keras.layers as tkl
import tensorflow as tf

class CALayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 temp,
                 rate=0.1):
        super(CALayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=d_model,
                                      num_heads=num_heads,
                                      temp=temp)
        self.mlp_block = MLPBlock(d_model=d_model,
                                  dff=dff,
                                  rate=rate)
        self.ln1 = tkl.LayerNormalization(epsilon=1e-6)
        self.ln2 = tkl.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tkl.Dropout(rate)
        self.dropout2 = tkl.Dropout(rate)

    def call(self, query, key, mask=None, training=None):
        mha_query, att = self.mha(value=key,
                                  key=key,
                                  query=query,
                                  mask=mask)
        mha_query = self.dropout1(mha_query, training=training)
        query = self.ln1(query + mha_query)

        mlp_query = self.mlp_block(query, training=training)
        mlp_query = self.dropout2(mlp_query, training=training)
        query = self.ln2(query + mlp_query)
        return query, att

class CALayers(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 temp,
                 rate):
        super(CALayers, self).__init__()
        self.num_layers = num_layers
        self.ca_layers = [CALayer(d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  temp=temp,
                                  rate=rate)
                          for _ in range(num_layers)]

    def call(self, query, key, mask=None, training=None, **kwargs):
        atts = []
        for i in range(self.num_layers):
            query, att = self.ca_layers[i](query=query,
                                           key=key,
                                           mask=mask,
                                           training=training)
            atts.append(att)
        return query, atts


class MCALayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 temps,
                 rate=0.1):
        """
        Implement multi cross attention layer.
        d_model (int): dimension in and out of the layer,
        num_heads (int): number of attention heads,
        dff (int): dimension of the projection inside the mlp block
        temps (list of int): List of temperatures to use for each input.
        """
        super(MCALayer, self).__init__()
        self.temps = temps
        self.n_ca = len(self.temps)
        self.mhas = [MultiHeadAttention(d_model=d_model,
                                        num_heads=num_heads,
                                        temp=temps[i])
                     for i in range(self.n_ca)]
        self.mhas_layernorm = [tkl.LayerNormalization(epsilon=1e-6) for _ in range(self.n_ca)]
        self.mhas_dropout = [tkl.Dropout(rate) for _ in range(self.n_ca)]

        self.mlp_block = MLPBlock(d_model=d_model,
                                  dff=dff,
                                  rate=rate)
        self.mlp_dropout = tkl.Dropout(rate) 
        self.mlp_layernorm = tkl.LayerNormalization(epsilon=1e-6)

    def call(self, query, keys, masks, training=None):
        """
        query (B, q, d_model): input queries
        keys (list of (B, k_i, d_model)) contain the input keys and values for each cross attention layer.
        masks (list of (B, q, k_i)): list of matrices of query authorizations for each cross attention_layer.
        """
        atts = []
        for i in range(self.n_ca):
            mha_query, att = self.mhas[i](value=keys[i],
                                          key=keys[i],
                                          query=query,
                                          mask=masks[i])
            atts.append(att)
            mha_query = self.mhas_dropout[i](mha_query, training=training)
            query = self.mhas_layernorm[i](query + mha_query)


        mlp_query = self.mlp_block(query, training=training)
        mlp_query = self.mlp_dropout(mlp_query, training=training)
        query = self.mlp_layernorm(query + mlp_query)
        return query, atts

class MCALayers(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 temps,
                 rate=0.1):

        super(MCALayers, self).__init__()
        self.mca_layers = [MCALayer(d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    temps=temps,
                                    rate=rate)
                           for _ in range(num_layers)]
        self.num_layers = num_layers
        self.layernorm = tkl.LayerNormalization(epsilon=1e-6)

    def call(self, query, keys, masks, training=None, **kwargs):
        """
        query (B, q, d_model): input queries
        keys (list of (B, k_i, d_model)) contain the input keys and values for each cross attention layer.
        masks (list of (B, q, k_i)): list of matrices of query authorizations for each cross attention_layer.
        """
        atts = []
        for i in range(self.num_layers):
            query, att = self.mca_layers[i](query=query,
                                            keys=keys,
                                            masks=masks,
                                            training=training)
            atts.append(att)

        query = self.layernorm(query)
        return query, atts
