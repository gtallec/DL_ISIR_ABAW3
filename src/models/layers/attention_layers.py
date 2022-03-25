import tensorflow.keras.layers as tkl
import tensorflow as tf

import math

class PatchEncoder(tkl.Layer):
    def __init__(self, num_patches, projection_dim, name='patch_encoder'):
        super(PatchEncoder, self).__init__(name=name)
        self.num_patches = num_patches
        self.projection = tkl.Dense(units=projection_dim)
        self.position_embedding = tkl.Embedding(input_dim=num_patches,
                                                output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class Patch_Extractor(tkl.Layer):
    def __init__(self, patch_size, name='patch_extractor'):
        super(Patch_Extractor, self).__init__(name=name)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class EncoderLayer(tkl.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = tkl.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)

        self.dense_relu = tkl.Dense(units=dff, activation='relu')
        self.dense_linear = tkl.Dense(units=d_model)

        self.layernorm1 = tkl.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tkl.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tkl.Dropout(rate)
        self.dropout2 = tkl.Dropout(rate)

    def call(self, x, training, mask):
        # (B, N_patch, d_model)
        attn_output, _ = self.mha(value=x,
                                  key=x,
                                  query=x,
                                  attention_mask=mask,
                                  return_attention_scores=True)
        # (B, N_patch, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # (B, N_patch, d_model)
        out1 = self.layernorm1(x + attn_output)  

        # (B, N_patch, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out1))
        # (B, N_patch, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (B, N_patch, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  

        return out2


class DecoderLayer(tkl.Layer):
    def __init__(self, d_model, num_heads, dff, label_temperature=1.0, rate=0.1, use_encoder=True):
        super(DecoderLayer, self).__init__()

        self.mha1 = TemperedMultiHeadAttention(key_dim=d_model, num_heads=num_heads, attention_axes=1, T=label_temperature)
        self.mha2 = tkl.MultiHeadAttention(key_dim=d_model, num_heads=num_heads, attention_axes=1)
 
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


    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        x :(B, T, d_model) encoded sequence from target sequence,
        enc_output (B, N_patch, d_model) : encoded sequence from src sequence,
        look_ahead_mask (T, T): label order mask, element of the sequence interact with previous elements only,
        padding_mask: Not relevant.
        """
        # (B, T, d_model)
        attn1, attn_weights_block1 = self.mha1(value=x,
                                               key=x,
                                               query=x,
                                               attention_mask=look_ahead_mask,
                                               return_attention_scores=True)
        # (B, T, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # (B, T, d_model), (B, H, T, N_patch)
        attn2, attn_weights_block2 = self.mha2(value=enc_output,
                                               key=enc_output,
                                               query=out1,
                                               attention_mask=padding_mask,
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


class Encoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 num_patches,
                 patch_size,
                 rate=0.1,
                 use_encoder=True):

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.patch_extractor = Patch_Extractor(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tkl.Dropout(rate)

    def call(self, x, training, mask):
        x = self.patch_extractor(x)
        x = self.patch_encoder(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x


class Decoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 label_temperature=1.0,
                 rate=0.1,
                 use_encoder=True):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.embedding_dense = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=embedding_init)

        self.position_embedding = tkl.Embedding(input_dim=T,
                                                output_dim=self.d_model)

        self.dec_layers = [DecoderLayer(d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        rate=rate,
                                        label_temperature=label_temperature,
                                        use_encoder=use_encoder)
                           for _ in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        x of size (B, T) sequence of labels shifted right.
        enc_output (B, N_patch, d_model)
        """
        output_dict = dict({})
     
        # Input encoding:
        # (1, T, T)
        encoding_matrix = tf.concat([tf.zeros((1, self.T)),
                                     tf.concat([tf.eye(self.T - 1),
                                                tf.zeros((self.T - 1, 1))], axis=1)],
                                    axis=0)[tf.newaxis, :, :]
        # (B, T, T)
        x = x[:, :, tf.newaxis] * encoding_matrix
        # (B, T, d_model)
        x = self.embedding_dense(x)

        # Compute distances between embedding vectors :
        # (T, d_model)
        tokens = self.embedding_dense.kernel
        # (T, )
        token_norms = tf.math.reduce_sum(tf.math.pow(tokens, 2), axis=1)
        # (T, T)
        dist_matrix = token_norms[:, tf.newaxis] + token_norms[tf.newaxis, :] - 2 * tf.matmul(tokens, tokens,
                                                                                              transpose_b=True)
        output_dict['token_dist_matrix'] = dist_matrix 

        # (B, T, d_model)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x=x,
                                                   enc_output=enc_output,
                                                   training=training,
                                                   look_ahead_mask=look_ahead_mask,
                                                   padding_mask=padding_mask)
            output_dict['dec_layer{}_block1'.format(i)] = block1
            output_dict['dec_layer{}_block2'.format(i)] = block2
        output_dict['x'] = x

        return output_dict


class MultiTaskDecoderLayer(tkl.Layer):
    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 use_encoder=True,
                 use_labels=True):
        super(MultiTaskDecoderLayer, self).__init__()

        self.mha_ty = TemperedMultiHeadAttention(key_dim=d_model,
                                                 num_heads=num_heads,
                                                 attention_axes=1,
                                                 temp=temp_ty)
        self.mha_tx = TemperedMultiHeadAttention(key_dim=d_model,
                                                 num_heads=num_heads,
                                                 attention_axes=1,
                                                 temp=temp_tx)
 
        self.dense_relu = tkl.Dense(units=dff, activation='relu')
        self.dense_linear = tkl.Dense(units=d_model)

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


    def call(self, x, y, tokens, look_ahead_mask, training):
        """
        x :(B, k_x, d_model): encoded keys from input,
        y :(B, T + 1, d_model): encoded sequence from output,
        tokens: (B, T, d_model): encoded tokens for each queried task 
        look_ahead_mask (T, T + 1): matrix of query authorization.
        """
        # (B, T, d_model)
        attn1, attn_weights_block1 = self.mha_ty(value=y,
                                                 key=y,
                                                 query=tokens,
                                                 attention_mask=look_ahead_mask,
                                                 return_attention_scores=True)
        # (B, T, d_model)
        attn1 = self.dropout_ty(attn1, training=training)
        out1 = self.layernorm_ty(self.labels_gate * attn1 + tokens)

        # (B, T, d_model), (B, H, T, N_patch)
        attn2, attn_weights_block2 = self.mha_tx(value=x,
                                                 key=x,
                                                 query=out1,
                                                 attention_mask=None,
                                                 return_attention_scores=True)
        # (B, T, d_model)
        attn2 = self.dropout_tx(attn2, training=training)
        # (B, T, d_model)
        out2 = self.layernorm_tx(self.encoder_gate * attn2 + out1)

        # (B, T, d_model)
        ffn_output = self.dense_linear(self.dense_relu(out2))
        # (B, T, d_model)
        ffn_output = self.dropout_tt(ffn_output, training=training)
        # (B, T, d_model)
        out3 = self.layernorm_tt(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

class MultiTaskDecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 use_encoder=True,
                 use_labels=True):
        super(MultiTaskDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.token_embedding = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=embedding_init)

        self.y_embedding = tkl.Dense(units=self.d_model,
                                     activation='linear',
                                     kernel_initializer=embedding_init)

        self.dec_layers = [MultiTaskDecoderLayer(d_model=d_model,
                                                 num_heads=num_heads,
                                                 dff=dff,
                                                 T=T,
                                                 rate=rate,
                                                 temp_ty=temp_ty,
                                                 temp_tx=temp_tx,
                                                 use_encoder=use_encoder,
                                                 use_labels=use_labels)
                           for _ in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T

    def call(self, x, y, tokens, look_ahead_mask, training):
        """
        x (B, N_patch, d_model) : Input encoded.
        y (B, T + 1) : Output Sequence.
        tokens (1, T, T): task queries.
        look_ahead_mask (1, T): matrix of authorization for tokens/y attention.
        """
        output_dict = dict({})
        B = tf.shape(x)[0]
     
        # # Output encoding:
        # (T + 1, T)
        encoding_matrix = tf.concat([tf.zeros((1, self.T)), tf.eye(self.T)],
                                    axis=0)

        # (B, T + 1, T)
        y = y[:, :, tf.newaxis] * encoding_matrix[tf.newaxis, :]
        # (B, T + 1, d_model)
        y = self.y_embedding(y)

        # # Tokens encoding
        # (1, T, d_model)
        tokens = self.token_embedding(tokens)
        tokens = tf.tile(tokens, multiples=(B, 1, 1))

        # (B, T, d_model)
        tokens = self.dropout(tokens, training=training)

        for i in range(self.num_layers):
            tokens, block_ty, block_tx = self.dec_layers[i](x=x,
                                                            y=y,
                                                            tokens=tokens,
                                                            training=training,
                                                            look_ahead_mask=look_ahead_mask)
            output_dict['dec_layer{}_block_ty'.format(i)] = block_ty
            output_dict['dec_layer{}_block_tx'.format(i)] = block_tx
        output_dict['x'] = tokens
        return output_dict
class UncertaintyDecoder(tkl.Layer):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 rate=0.1):
        super(UncertaintyDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.embedding_dense = tkl.Dense(units=self.d_model,
                                         activation='linear',
                                         kernel_initializer=embedding_init)


        self.dec_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tkl.Dropout(rate)
        self.T = T

    def call(self, x, training, look_ahead_mask):
        """
        x of size (B, T) sequence of labels shifted right.
        enc_output (B, N_patch, d_model)
        """
     
        # Input encoding:
        # (1, T, T)
        encoding_matrix = tf.concat([tf.zeros((1, self.T)),
                                     tf.concat([tf.eye(self.T - 1),
                                                tf.zeros((self.T - 1, 1))], axis=1)],
                                    axis=0)[tf.newaxis, :, :]
        # (B, T, T)
        x = x[:, :, tf.newaxis] * encoding_matrix
        # (B, T, d_model)
        x = self.embedding_dense(x)
        # (B, T, d_model)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x=x,
                                   training=training,
                                   mask=look_ahead_mask)
        return x


class TemperedMultiHeadAttention(tkl.MultiHeadAttention):
    def __init__(self,
                 num_heads,
                 key_dim,
                 temp=1.0,
                 value_dim=None,
                 dropout=0.0,
                 use_bias=True,
                 output_shape=None,
                 attention_axes=None,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TemperedMultiHeadAttention, self).__init__(num_heads,
                                                         key_dim=key_dim,
                                                         value_dim=value_dim,
                                                         dropout=dropout,
                                                         use_bias=use_bias,
                                                         output_shape=output_shape,
                                                         attention_axes=attention_axes,
                                                         kernel_initializer=kernel_initializer,
                                                         bias_initializer=bias_initializer,
                                                         kernel_regularizer=kernel_regularizer,
                                                         bias_regularizer=bias_regularizer,
                                                         activity_regularizer=activity_regularizer,
                                                         kernel_constraint=kernel_constraint,
                                                         bias_constraint=bias_constraint,
                                                         **kwargs)
        self.temp = temp

    def _compute_attention(self,
                           query,
                           key,
                           value,
                           attention_mask=None,
                           training=None):
        """Applies Dot-product attention with query, key, value tensors.
        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for customized
        attention implementation.
        Args:
        query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
        key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
        value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
        attention_mask: a boolean mask of shape `(B, T, S)`, that prevents attention to certain positions.
        training: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing).
        Returns:
        attention_output: Multi-headed outputs of attention computation.
        attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        # (B, N, Q, d_model)
        tf.print("query inside shape : ", tf.shape(query))

        tf.print("query inside mean : ", tf.math.reduce_mean(query, axis=-1))
        tf.print("query inside std : ", tf.math.reduce_std(query, axis=-1))
        query = tf.multiply(query, 1.0 / (math.sqrt(float(self._key_dim)) * self.temp))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # (B, N, H, Q, K)
        attention_scores = tf.einsum(self._dot_product_equation, key,
                                     query)
        
        # tf.print("attention_logits mean : ", tf.math.reduce_mean(attention_scores, axis=-1))
        # tf.print("attention_logits std : ", tf.math.reduce_std(attention_scores, axis=-1))

        # (B, N, H, Q, K)
        # tf.print("attention_mask : ", attention_mask)
        attention_scores = self._masked_softmax(attention_scores, attention_mask) 
        # tf.print("attention_scores mean : ", tf.math.reduce_mean(attention_scores, axis=-1))
        # tf.print("attention_scores std : ", tf.math.reduce_std(attention_scores, axis=-1))

        # tf.print(50 * '#')
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(self._combine_equation,
                                     attention_scores_dropout, value)
        return attention_output, attention_scores
