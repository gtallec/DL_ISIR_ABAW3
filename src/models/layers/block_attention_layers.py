import numpy as np
import tensorflow.keras.layers as tkl
import tensorflow as tf


from models.layers.attention_layers import TemperedMultiHeadAttention

def generate_block_look_ahead_masks(orders, blocksize):
    """ Generate the look ahead masks for M different orders of T tasks
    Input : 
    orders of size (M, T).
    blocksize (int): size of the blocks of task
    Output : Look ahead mask of size (M, T)
    """
    P, T = orders.shape
    look_ahead_masks = np.zeros((P, T, T + 1))
    for p in range(P):
        look_ahead_masks[p, :, :] = generate_block_look_ahead_mask(orders[p, :], blocksize)
    return look_ahead_masks

def generate_block_look_ahead_mask(order, blocksize):
    T = order.shape[0]
    block_assignment = order.reshape((blocksize, -1))
    look_ahead_mask = np.concatenate([np.ones((T, 1)), np.zeros((T, T))], axis=1)
    for t in range(T):
        block_number, _ = np.where(block_assignment == t)
        sliced_block_assignement = block_assignment[:block_number[0], :]
        for block_id in range(sliced_block_assignement.shape[0]):
            for task_id in sliced_block_assignement[block_id, :]:
                look_ahead_mask[t, task_id + 1] = 1
    return look_ahead_mask
            

class BMOMTDecoderLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 orders,
                 blocksize,
                 controller,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1):
        super(BMOMTDecoderLayer, self).__init__()

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
        self.controller = controller
        # (M, T)
        self.orders = orders
        # (M, T, T + 1)
        self.look_ahead_masks = tf.Variable(generate_block_look_ahead_masks(self.orders, blocksize),
                                            dtype=tf.int32,
                                            trainable=False)

        self.layernorm_ty = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_tx = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm_tt = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout_ty = tkl.Dropout(rate)
        self.dropout_tx = tkl.Dropout(rate)
        self.dropout_tt = tkl.Dropout(rate)


    def call(self, x, y, tokens, order_indices, training):
        """
        x :(?, k_x, d_model): encoded keys from input,
        y :(?, T + 1, d_model): encoded sequence from output,
        tokens: (?, T, d_model): encoded tokens for each queried task
        order_indices (?, ) : Orders to use for each element of the batch
        """
        # (?, 1, T, T + 1)
        selected_look_ahead_masks = tf.gather(tf.expand_dims(self.look_ahead_masks, axis=1),
                                              order_indices,
                                              axis=0)

        # (?, T, d_model), (?, H, T, T + 1)
        attn_ty, attn_weights_ty = self.mha_ty(value=y,
                                               key=y,
                                               query=tokens,
                                               attention_mask=selected_look_ahead_masks,
                                               return_attention_scores=True)
        # (?, T, d_model)
        attn_ty = self.dropout_ty(attn_ty, training=training)
        # (?, T, d_model)
        out_ty = self.layernorm_ty(attn_ty + tokens)

        # (?, T, d_model), (?, H, T, N_patch)
        attn_tx, attn_weights_tx = self.mha_tx(value=x,
                                               key=x,
                                               query=out_ty,
                                               attention_mask=None,
                                               return_attention_scores=True)
        # (?, T, d_model)
        attn_tx = self.dropout_tx(attn_tx, training=training)
        # (?, T, d_model)
        controlled_attn_tx = self.controller(attn_tx)
        out_tx = self.layernorm_tx(controlled_attn_tx + out_ty)

        # (?, T, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out_tx))
        # (?, T, d_model)
        ffn_output = self.dropout_tt(ffn_output, training=training)
        # (?, T, d_model)
        out_tt = self.layernorm_tt(ffn_output + out_tx)
        return out_tt, attn_weights_ty, attn_weights_tx


class BMOMTDecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 controllers,
                 T,
                 orders,
                 blocksize,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 permutation_encoding=False):
        super(BMOMTDecoder, self).__init__()
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

        self.dec_layers = [BMOMTDecoderLayer(d_model=d_model,
                                             num_heads=num_heads,
                                             dff=dff,
                                             orders=orders,
                                             blocksize=blocksize,
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
