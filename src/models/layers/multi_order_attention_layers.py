import numpy as np
import tensorflow.keras.layers as tkl
import tensorflow as tf


from models.layers.attention_layers import TemperedMultiHeadAttention

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

class MOMTDecoderLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 orders,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 use_encoder=True,
                 use_labels=True):
        super(MOMTDecoderLayer, self).__init__()

        self.mha_ty = TemperedMultiHeadAttention(key_dim=d_model,
                                                 num_heads=num_heads,
                                                 attention_axes=-3,
                                                 temp=temp_ty)
        self.mha_tx = TemperedMultiHeadAttention(key_dim=d_model,
                                                 num_heads=num_heads,
                                                 attention_axes=-3,
                                                 temp=temp_tx)
 
        self.dense_relu = tkl.Dense(units=dff, activation='relu')
        self.dense_linear = tkl.Dense(units=d_model)
        # (M, T)
        self.orders = orders
        # (M, T, T + 1)
        self.look_ahead_masks = tf.Variable(generate_order_look_ahead_masks(self.orders),
                                            dtype=tf.int32,
                                            trainable=False)

        self.layernorm_ty = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_tx = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_tt = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout_ty = tkl.Dropout(rate)
        self.dropout_tx = tkl.Dropout(rate)
        self.dropout_tt = tkl.Dropout(rate)

        if use_encoder:
            self.encoder_gate = tf.constant(1, dtype=tf.float32)
        else:
            self.encoder_gate = tf.constant(0, dtype=tf.float32)

        if use_labels:
            self.labels_gate = tf.constant(1, dtype=tf.float32)
        else:
            self.labels_gate = tf.constant(0, dtype=tf.float32)


    def call(self, x, y, tokens, order_indices, training):
        """
        x :(?, k_x, d_model): encoded keys from input,
        y :(?, T + 1, d_model): encoded sequence from output,
        tokens: (?, T, d_model): encoded tokens for each queried task
        order_indices (?, ) : Orders to use for each element of the batch
        """
        # (?, 1, T, T + 1) -> (?, H, T, T + 1)
        selected_look_ahead_masks = tf.gather(self.look_ahead_masks,
                                              order_indices,
                                              axis=-3)[:, tf.newaxis, :, :]

        # (?, T, d_model), (?, H, T, T + 1)
        attn_ty, attn_weights_ty = self.mha_ty(value=y,
                                               key=y,
                                               query=tokens,
                                               attention_mask=selected_look_ahead_masks,
                                               return_attention_scores=True)
        # (?, T, d_model)
        attn_ty = self.dropout_ty(attn_ty, training=training)
        # (?, T, d_model)
        out_ty = self.layernorm_ty(self.labels_gate * attn_ty + tokens)

        # (?, T, d_model), (?, H, T, N_patch)
        attn_tx, attn_weights_tx = self.mha_tx(value=x,
                                               key=x,
                                               query=out_ty,
                                               attention_mask=None,
                                               return_attention_scores=True)
        # (?, T, d_model)
        attn_tx = self.dropout_tx(attn_tx, training=training)
        # (?, T, d_model)
        out_tx = self.layernorm_tx(self.encoder_gate * attn_tx + out_ty)

        # (?, T, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out_tx))
        # (?, T, d_model)
        ffn_output = self.dropout_tt(ffn_output, training=training)
        # (?, T, d_model)
        out_tt = self.layernorm_tt(ffn_output + out_tx)
        return out_tt, attn_weights_ty, attn_weights_tx

class MOMTDecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 orders,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 use_encoder=True,
                 use_labels=True,
                 permutation_encoding=False):
        super(MOMTDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.token_embedding = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=embedding_init)

        self.y_embedding = tkl.Dense(units=self.d_model,
                                     activation='linear',
                                     kernel_initializer=embedding_init)


        self.permutation_encoding = permutation_encoding
        if self.permutation_encoding:
            self.permutation_embedding = tkl.Dense(units=self.d_model,
                                                   activation='linear',
                                                   kernel_initializer=embedding_init)
            self.permutation_tokens = tf.Variable(tf.eye(tf.shape(orders)[0]),
                                                  dtype=tf.float32,
                                                  trainable=False)

        self.dec_layers = [MOMTDecoderLayer(d_model=d_model,
                                            num_heads=num_heads,
                                            dff=dff,
                                            orders=orders,
                                            rate=rate,
                                            temp_ty=temp_ty,
                                            temp_tx=temp_tx,
                                            use_encoder=use_encoder,
                                            use_labels=use_labels)
                           for _ in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T
        self.permutation_encoding = permutation_encoding

    def call(self, x, y, tokens, order_indices, training):
        """
        x (?, N_patch, d_model) : Input encoded.
        y (?, T + 1) : Output Sequence.
        tokens (1, T, T): task queries.
        look_ahead_mask (1, T): matrix of authorization for tokens/y attention.
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
        # (?, T + 1, d_model)
        y = self.y_embedding(y)

        # # Tokens encoding
        # (1, 1, T, d_model)
        tokens = self.token_embedding(tokens)
        # (?, T, d_model)
        tokens = tf.tile(tokens, multiples=tf.concat([B, [1, 1]], axis=0))

        if self.permutation_encoding:
            # (B, 1, d_model)
            permutation_tokens = tf.gather(params=self.permutation_tokens,
                                           indices=order_indices,
                                           axis=0)[:, tf.newaxis, :]
            tokens = tokens + self.permutation_embedding(permutation_tokens,
                                                         training=training)

        # (B, T, d_model)
        tokens = self.dropout(tokens, training=training)

        for i in range(self.num_layers):
            tokens, block_ty, block_tx = self.dec_layers[i](x=x,
                                                            y=y,
                                                            tokens=tokens,
                                                            order_indices=order_indices,
                                                            training=training)
            output_dict['dec_layer{}_block_ty'.format(i)] = block_ty
            output_dict['dec_layer{}_block_tx'.format(i)] = block_tx
        output_dict['x'] = tokens
        return output_dict

class MOMTDecoderWithController(tkl.Layer):
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
                 use_encoder=True,
                 use_labels=True,
                 permutation_encoding=False):
        super(MOMTDecoderWithController, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.token_embedding = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=embedding_init)

        self.y_embedding = tkl.Dense(units=self.d_model,
                                     activation='linear',
                                     kernel_initializer=embedding_init)


        self.permutation_encoding = permutation_encoding
        if self.permutation_encoding:
            self.permutation_embedding = tkl.Dense(units=self.d_model,
                                                   activation='linear',
                                                   kernel_initializer=embedding_init)
            self.permutation_tokens = tf.Variable(tf.eye(tf.shape(orders)[0]),
                                                  dtype=tf.float32,
                                                  trainable=False)

        self.dec_layers = [MOMTDecoderWithControllerLayer(d_model=d_model,
                                                          num_heads=num_heads,
                                                          dff=dff,
                                                          orders=orders,
                                                          rate=rate,
                                                          temp_ty=temp_ty,
                                                          temp_tx=temp_tx,
                                                          controller=controllers.get(i),
                                                          use_labels=use_labels)
                           for i in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T
        self.permutation_encoding = permutation_encoding

    def call(self, x, y, tokens, order_indices, training):
        """
        x (?, N_patch, d_model) : Input encoded.
        y (?, T + 1) : Output Sequence.
        tokens (1, T, T): task queries.
        look_ahead_mask (1, T): matrix of authorization for tokens/y attention.
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
        # (?, T + 1, d_model)
        y = self.y_embedding(y)

        # # Tokens encoding
        # (1, 1, T, d_model)
        tokens = self.token_embedding(tokens)
        # (?, T, d_model)
        tokens = tf.tile(tokens, multiples=tf.concat([B, [1, 1]], axis=0))

        if self.permutation_encoding:
            # (B, 1, d_model)
            permutation_tokens = tf.gather(params=self.permutation_tokens,
                                           indices=order_indices,
                                           axis=0)[:, tf.newaxis, :]
            tokens = tokens + self.permutation_embedding(permutation_tokens,
                                                         training=training)

        # (B, T, d_model)
        tokens = self.dropout(tokens, training=training)

        for i in range(self.num_layers):
            tokens, block_ty, block_tx = self.dec_layers[i](x=x,
                                                            y=y,
                                                            tokens=tokens,
                                                            order_indices=order_indices,
                                                            training=training)
            output_dict['dec_layer{}_block_ty'.format(i)] = block_ty
            output_dict['dec_layer{}_block_tx'.format(i)] = block_tx
        output_dict['x'] = tokens
        return output_dict



class MultiOrderDecoderLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 orders,
                 y_temperature=1.0,
                 xy_temperature=1.0,
                 rate=0.1,
                 use_encoder=True):

        super(MultiOrderDecoderLayer, self).__init__()

        self.mha1 = TemperedMultiHeadAttention(key_dim=d_model, num_heads=num_heads, attention_axes=1, T=y_temperature)
        self.mha2 = TemperedMultiHeadAttention(key_dim=d_model, num_heads=num_heads, attention_axes=1, T=xy_temperature)
        # (M, T)
        self.orders = orders
        orders_shape = tf.shape(orders)
        M = orders_shape[0]
        self.T = orders_shape[1]
        order_matrices = tf.gather(tf.eye(T), orders, axis=0)
        invert_order_matrices = tf.transpose(order_matrices, perm=(0, 2, 1))
        # (M, T)
        self.invert_order = tf.math.reduce_sum(invert_order_matrices * tf.tile(tf.range(T)[tf.newaxis, tf.newaxis, :],
                                                                               (M, T, 1)),
                                               axis=1)

        self.dense_relu = tkl.Dense(units=dff, activation='relu')
        self.dense_linear = tkl.Dense(units=d_model)

        self.layernorm1 = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tkl.Dropout(rate)
        self.dropout2 = tkl.Dropout(rate)
        self.dropout3 = tkl.Dropout(rate)

        if use_encoder:
            self.encoder_gate = tf.constant(1, dtype=tf.float32)
        else:
            self.encoder_gate = tf.constant(0, dtype=tf.float32)


    def call(self, x, enc_output, training, order_indices):
        """
        x (B, T + 1, d_model) : encoded labels shifted right.
        enc_output (B, N_patch, d_model) : encoded image,
        look_ahead_mask (B, T, T + 1): order based masks 
        """
        # (B, T)
        B = tf.shape(x)[0]
        selected_orders = tf.gather(self.orders,
                                    order_indices,
                                    axis=0)
        # (B, T + 1)
        padded_selected_orders = tf.concat([tf.zeros((B, 1)), selected_orders + 1],
                                           axis=1)

        # Permute keys and queries : 
        # (B, T + 1, T)
        x = tf.gather(x,
                      padded_selected_orders,
                      batch_dims=1,
                      axis=1)

        look_ahead_mask = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0)[tf.newaxis, tf.newaxis, :] 
        # (B, T, d_model)
        # attn1 (B, T, d_model), attn_weights_block1 (B, T, T)
        attn1, attn_weights_block1 = self.mha1(value=x[:, :-1, :],
                                               key=x[:, :-1, :],
                                               query=x[:, 1:, :],
                                               attention_mask=look_ahead_mask,
                                               return_attention_scores=True)
        # (B, T, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
 
        # attn2 (B, T, d_model), attn_weights_block2 (B, T, T + 1)
        attn2, attn_weights_block2 = self.mha2(value=enc_output,
                                               key=enc_output,
                                               query=out1,
                                               return_attention_scores=True)
        # (B, T, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (B, T, d_model)
        out2 = self.layernorm2(self.encoder_gate * attn2 + out1)

        # (B, T, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out2))
        # (B, T, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (B, T, d_model)
        out3 = self.layernorm3(ffn_output + out2)  
        return out3, attn_weights_block1, attn_weights_block2


class MultiOrderDecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 orders,
                 y_temperature=1.0,
                 xy_temperature=1.0,
                 rate=0.1,
                 use_encoder=True):
        super(MultiOrderDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        P = orders.shape[0]

        label_embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        order_embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        self.label_embedding = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=label_embedding_init,
                                         use_bias=False)

        self.order_embedding = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=order_embedding_init,
                                         use_bias=False)

        self.dec_layers = [MultiOrderDecoderLayer(d_model=d_model,
                                                  num_heads=num_heads,
                                                  dff=dff,
                                                  orders=orders,
                                                  rate=rate,
                                                  y_temperature=y_temperature,
                                                  xy_temperature=xy_temperature,
                                                  use_encoder=use_encoder)
                           for _ in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T

        # (P, T)
        self.order_tokens = tf.eye(P)

    def call(self, x, enc_output, training, order_indices):
        """
        x (B, T): labels.
        enc_output (B, N_patch, d_model): Encoded input.
        order_indices (B, ): index of the order to use for each element.
        """
        B = tf.shape(x)[0]
        order_tokens = tf.gather(self.order_tokens,
                                 indices=order_indices,
                                 axis=0)

        output_dict = dict({}) 
        # Label position encoding:
        # (B, T, T)
        x = x[:, :, tf.newaxis] * tf.eye(self.T)

        # (B, T + 1, T)
        x = tf.concatenate([tf.zeros((B, 1, self.T)), x],
                           axis=1)

        # (B, T + 1, d_model)
        x = self.label_embedding(x)

        # (B, P)
        # (B, 1, d_model)
        encoded_order_tokens = self.order_embedding(order_tokens)[:, tf.newaxis, :]

        # (B, T + 1, d_model)
        x = self.dropout(x + encoded_order_tokens, training=training)
 
        # (1, 1, T, T) 
        look_ahead_mask = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0)[tf.newaxis, tf.newaxis, :, :]
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x=x,
                                                   enc_output=enc_output,
                                                   order_indices=order_indices,
                                                   training=training,
                                                   look_ahead_mask=look_ahead_mask)
            output_dict['dec_layer{}_block1'.format(i)] = block1
            output_dict['dec_layer{}_block2'.format(i)] = block2
        output_dict['x'] = x

        return output_dict
