import tensorflow.keras.layers as tkl
import tensorflow as tf

from models.layers.attention_modules import MultiHeadAttention

class LLADecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 attention_mode,
                 controllers,
                 T,
                 temp_yy=1.0,
                 temp_yx=1.0,
                 rate=0.1):
        super(LLADecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        self.y_embedding = tkl.Dense(units=self.d_model,
                                     activation='linear',
                                     kernel_initializer=embedding_init,
                                     use_bias=False)

        self.dec_layers = [LLADecoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dff=dff,
                                           attention_mode=attention_mode,
                                           controller=controllers.get(i),
                                           temp_yy=temp_yy,
                                           temp_yx=temp_yx,
                                           rate=rate)
                           for i in range(num_layers)]

        self.dropout = tkl.Dropout(rate)
        self.T = T

    def call(self, x, y, look_ahead_mask, training):
        """
        x (B, N, N_patch, d_model) : Input encoded.
        y (B, N, T + 1) : Output Sequence.
        look_ahead_mask (B, N, 1, T, T): matrix of authorization for y/y attention.
        """
        output_dict = dict({})
        B = tf.shape(y)[:-1]

        # # Output encoding:
        # (T + 1, T)
        encoding_matrix = tf.concat([tf.zeros((1, self.T)), tf.eye(self.T)],
                                    axis=0)
        # (ones_like(?), T + 1, T)
        encoding_matrix = tf.reshape(encoding_matrix, tf.concat([tf.ones_like(B), tf.shape(encoding_matrix)],
                                                                axis=0)) 

        # (?, T + 1, T)
        y = tf.expand_dims(y, axis=-1) * encoding_matrix
        # (?, T, T)
        y = y[:, :, :-1, :]
        # (?, T, d_model)
        y = self.y_embedding(y)
        # (?, T, d_model)
        y = self.dropout(y, training=training)
        query_y = y
        # tf.print("query y_corr :", tf.matmul(query_y, query_y, transpose_b=True) / self.d_model)
        for i in range(self.num_layers):
            # tf.print("Layer {}".format(i))
            # tf.print("query y_corr :", tf.matmul(query_y, query_y, transpose_b=True) / self.d_model)
            query_y, block_yy, block_yx = self.dec_layers[i](x=x,
                                                             query_y=query_y,
                                                             y=y,
                                                             look_ahead_mask=look_ahead_mask,
                                                             training=training)
            output_dict['dec_layer{}_block_yy'.format(i)] = block_yy
            output_dict['dec_layer{}_block_yx'.format(i)] = block_yx
        output_dict['x'] = query_y

        return output_dict

class LLADecoderLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 attention_mode,
                 controller,
                 temp_yy=1.0,
                 temp_yx=1.0,
                 rate=0.1):
        super(LLADecoderLayer, self).__init__()

        self.mha_yy = MultiHeadAttention(d_model=d_model,
                                         num_heads=num_heads,
                                         temp=temp_yy)
        self.mha_yx = MultiHeadAttention(d_model=d_model,
                                         num_heads=num_heads,
                                         temp=temp_yx)
        self.attention_mode = attention_mode
        self.d_model = d_model

        self.dense_relu = tkl.Dense(units=dff, activation='relu')
        self.dense_linear = tkl.Dense(units=d_model)
        self.controller = controller

        self.layernorm_yy = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_yx = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout_yy = tkl.Dropout(rate)
        self.dropout_yx = tkl.Dropout(rate)
        self.dropout = tkl.Dropout(rate)

    def call(self, x, query_y, y, look_ahead_mask, training):
        """
        x :(B, N, k_x, d_model): encoded keys from input,
        y :(B, N, T, d_model): encoded sequence from output,
        look_ahead_mask (B, N, T, T): matrix of query authorization.
        """
        # (B, N, T, d_model)
        k_x = tf.shape(x)[-2]
        T = tf.shape(y)[-2]

        if self.attention_mode == 'self':
            key_y = query_y
            value_y = query_y
        elif self.attention_mode == 'cross':
            key_y = y
            value_y = y

        attn1, attn_weights_block1 = self.mha_yy(value=value_y,
                                                 key=key_y,
                                                 query=query_y,
                                                 mask=look_ahead_mask)
        # (B, N, T, d_model)
        attn1 = self.dropout_yy(attn1, training=training)
        out1 = self.layernorm_yy(attn1 + query_y)

        # (B, N, T, d_model), (B, N, H, T, N_patch)
        attn2, attn_weights_block2 = self.mha_yx(value=x,
                                                 key=x,
                                                 query=out1,
                                                 mask=tf.ones((1, 1, 1, T, k_x)))
        # (B, N, T, d_model)
        attn2 = self.dropout_yx(attn2, training=training)
        # (B, N, T, d_model)
        out2 = self.layernorm_yx(self.controller(attn2) + out1)

        # (B, N, T, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out2))
        # (B, N, T, d_model)
        ffn_output = self.dropout(ffn_output, training=training)
        # (B, N, T, d_model)
        out3 = self.layernorm(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

