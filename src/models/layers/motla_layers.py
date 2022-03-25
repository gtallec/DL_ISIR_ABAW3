import numpy as np
import tensorflow.keras.layers as tkl
import tensorflow as tf


from models.layers.attention_modules import MultiHeadAttention

def generate_order_look_ahead_masks(orders):
    """ Generate the look ahead masks for M different orders of T tasks
    Input : orders of size (M, T).
    Output : Look ahead mask of size (M, T)
    """
    P, T = orders.shape
    look_ahead_masks = np.zeros((P, T, T + 1))
    for t in range(T):
        mat = np.concatenate([np.ones((P, 1)), np.zeros((P, T))],
                             axis=1)
        for m in range(P):
            i = 0
            while (orders[m, i] != t) and (i < T):
                available_key = orders[m, i]
                mat[m, 1 + available_key] = 1
                i += 1
        look_ahead_masks[:, t, :] = mat

    return look_ahead_masks

class MOTLADecoderLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 orders,
                 controller,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1):
        super(MOTLADecoderLayer, self).__init__()

        self.mha_ty = MultiHeadAttention(d_model=d_model,
                                         num_heads=num_heads,
                                         temp=temp_ty)
        self.mha_tx = MultiHeadAttention(d_model=d_model,
                                         num_heads=num_heads,
                                         temp=temp_tx)
 
        self.dense_relu = tkl.Dense(units=dff, activation='relu')
        self.dense_linear = tkl.Dense(units=d_model)
        self.controller = controller
        # (M, T)
        self.orders = orders
        # (M, T, T + 1)
        self.look_ahead_masks = tf.Variable(generate_order_look_ahead_masks(self.orders),
                                            dtype=tf.float32,
                                            trainable=False)

        self.layernorm_ty = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_tx = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_tt = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout_ty = tkl.Dropout(rate)
        self.dropout_tx = tkl.Dropout(rate)
        self.dropout_tt = tkl.Dropout(rate)


    def call(self, x, y, tokens, order_indices, training):
        """
        x :(B, N, k_x, d_model): encoded keys from input,
        y :(B, N, T + 1, d_model): encoded sequence from output,
        tokens: (B, N, T, d_model): encoded tokens for each queried task
        order_indices (B, N) : Orders to use for each element of the batch
        """
        T = tf.shape(tokens)[-2]
        k_x = tf.shape(x)[-2]
        # (B, N, 1, T, T + 1) -> (B, N, H, T, T + 1)
        selected_look_ahead_masks = tf.gather(tf.expand_dims(self.look_ahead_masks, axis=1),
                                              order_indices,
                                              axis=0)

        # (B, N, T, d_model), (B, N, H, T, T + 1)
        attn_ty, attn_weights_ty = self.mha_ty(value=y,
                                               key=y,
                                               query=tokens,
                                               mask=selected_look_ahead_masks)
        # (B, N, T, d_model)
        attn_ty = self.dropout_ty(attn_ty, training=training)
        # (B, N, T, d_model)
        out_ty = self.layernorm_ty(attn_ty + tokens)

        # (B, N, T, d_model), (B, N, H, T, N_patch)
        attn_tx, attn_weights_tx = self.mha_tx(value=x,
                                               key=x,
                                               query=out_ty,
                                               mask=tf.ones((1, 1, 1, T, k_x)))
        # (B, N, T, d_model)
        attn_tx = self.dropout_tx(attn_tx, training=training)
        # (B, N, T, d_model)
        controlled_attn_tx = self.controller(attn_tx)
        # (B, N, T, d_model)
        out_tx = self.layernorm_tx(controlled_attn_tx + out_ty)

        # (B, N, T, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out_tx))
        # (B, N, T, d_model)
        ffn_output = self.dropout_tt(ffn_output, training=training)
        # (B, N, T, d_model)
        out_tt = self.layernorm_tt(ffn_output + out_tx)
        return out_tt, attn_weights_ty, attn_weights_tx


class MOTLADecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 controllers,
                 T,
                 orders,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 permutation_encoding=False):
        super(MOTLADecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.token_embedding = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=embedding_init,
                                         use_bias=False)

        self.y_embedding = tkl.Dense(units=self.d_model,
                                     activation='linear',
                                     kernel_initializer=embedding_init,
                                     use_bias=False)


        self.permutation_encoding = permutation_encoding
        if self.permutation_encoding:
            self.permutation_embedding = tkl.Dense(units=self.d_model,
                                                   activation='linear',
                                                   kernel_initializer=embedding_init)
            self.permutation_tokens = tf.Variable(tf.eye(tf.shape(orders)[0]),
                                                  dtype=tf.float32,
                                                  trainable=False)

        self.dec_layers = [MOTLADecoderLayer(d_model=d_model,
                                             num_heads=num_heads,
                                             dff=dff,
                                             orders=orders,
                                             rate=rate,
                                             temp_ty=temp_ty,
                                             temp_tx=temp_tx,
                                             controller=controllers.get(i))
                           for i in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T
        self.permutation_encoding = permutation_encoding

    def call(self, x, y, tokens, order_indices, training):
        """
        x (B, N, N_patch, d_model) : Input encoded.
        y (B, N, T + 1) : Output Sequence.
        tokens (1, 1, T, T): task queries.
        look_ahead_mask (1, T): matrix of authorization for tokens/y attention.
        """
        output_dict = dict({})
        B = tf.shape(y)[:-1] 
        # # Output encoding:
        # (T + 1, T)
        encoding_matrix = tf.concat([tf.zeros((1, self.T)), tf.eye(self.T)],
                                    axis=0)
        # (B, N, T + 1, T)
        encoding_matrix = tf.reshape(encoding_matrix, tf.concat([tf.ones_like(B), tf.shape(encoding_matrix)],
                                                                axis=0)) 

        # (B, N, T + 1, T)
        y = tf.expand_dims(y, axis=-1) * encoding_matrix
        # (B, N, T + 1, d_model)
        y = self.y_embedding(y)

        # # Tokens encoding
        # (1, 1, T, d_model)
        tokens = self.token_embedding(tokens)
        # (B, N, T, d_model)
        tokens = tf.tile(tokens, multiples=tf.concat([B, [1, 1]], axis=0))

        if self.permutation_encoding:
            # (B, 1, d_model)
            permutation_tokens = tf.gather(params=self.permutation_tokens,
                                           indices=order_indices,
                                           axis=0)[:, tf.newaxis, :]
            tokens = tokens + self.permutation_embedding(permutation_tokens,
                                                         training=training)

        # (B, N, T, d_model)
        tokens = self.dropout(tokens, training=training)

        for i in range(self.num_layers):
            # (B, N, T, d_model)
            tokens, block_ty, block_tx = self.dec_layers[i](x=x,
                                                            y=y,
                                                            tokens=tokens,
                                                            order_indices=order_indices,
                                                            training=training)
            output_dict['dec_layer{}_block_ty'.format(i)] = block_ty
            output_dict['dec_layer{}_block_tx'.format(i)] = block_tx
        output_dict['x'] = tokens
        return output_dict
