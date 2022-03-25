import tensorflow as tf
import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm

def scaled_dot_product_attention(q, k, v, mask, temperature=1.0, epsilon=1e-6):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition. 
    In the mask: 1 means do attention 0 means mask attention

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., n_h, seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # (..., n_h, seq_len_q, seq_len_k)
    scaled_attention_logits = matmul_qk / (tf.math.sqrt(dk) * temperature)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (1 - mask) * -1e9


    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tkl.Layer):
    def __init__(self, d_model, num_heads, temp=1.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.dense = tkl.Dense(d_model, use_bias=True)

        self.temp = temp
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (... num_heads, seq_len, depth)
        """
        x = tf.reshape(x, tf.concat([batch_size, [-1, self.num_heads, self.depth]], axis=0))
        n = tf.shape(batch_size)[0]
        perm_axes = tf.concat([tf.range(n), [n + 1, n, n + 2]], axis=0)
        return tf.transpose(x, perm=perm_axes)

    def call(self, qkv, mask=None):
        """
        value (..., len_k, d_model)
        key (..., len_k, d_model)
        query (..., len_q, d_model)
        mask (..., len_q, len_k)
        """
        query, key, value = qkv
        batch_size = tf.shape(query)[:-2]
        len_k = tf.shape(key)[-2]
        len_q = tf.shape(query)[-2]
        ones_like_batchsize = tf.ones_like(batch_size)

        if mask is None:
            mask = tf.ones(tf.concat([ones_like_batchsize, [1, len_q, len_k]], axis=0))

        # bias_reshape = tf.concat([ones_like_batchsize, [1, self.d_model]], axis=0)

        query = self.wq(query) # + tf.reshape(self.bq, bias_reshape)
        # (..., seq_len, d_model)
        key = self.wk(key) # + tf.reshape(self.bk, bias_reshape)  # (..., seq_len, d_model)
        value = self.wv(value) # + tf.reshape(self.bv, bias_reshape)  # (..., seq_len, d_model)

        query = self.split_heads(query, batch_size)  # (..., num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (..., num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (..., num_heads, seq_len_v, depth)

        # (..., num_heads, seq_len_q, depth), (..., num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q=query,
                                                                           k=key,
                                                                           v=value,
                                                                           mask=mask,
                                                                           temperature=self.temp)
        n = tf.shape(batch_size)[0]
        perm_axes = tf.concat([tf.range(n), [n + 1, n, n + 2]], axis=0)

        # (..., seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=perm_axes)  
        # (..., seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, tf.concat([batch_size, [-1, self.d_model]], axis=0))  
        output = self.dense(concat_attention) # + tf.reshape(self.b, bias_reshape)  # (..., seq_len_q, d_model)
        return output, attention_weights

    def build(self, input_shape):
        q_shape, k_shape, v_shape = input_shape 
        self.wq.build(q_shape)
        self.wk.build(k_shape)
        self.wv.build(v_shape)
        self.dense.build(q_shape)
        super().build(input_shape)


class MLPBlock(tkm.Model):
    def __init__(self,
                 d_model,
                 dff,
                 rate):
        super(MLPBlock, self).__init__()
        self.dense_gelu = tkl.Dense(units=dff,
                                    activation='gelu')
        self.dropout1 = tkl.Dropout(rate=rate)
        self.dense_linear = tkl.Dense(units=d_model,
                                      activation='linear')
        self.dropout2 = tkl.Dropout(rate=rate)

    def call(self, x, training=None, **kwargs):
        x = self.dense_gelu(x)
        x = self.dropout1(x, training=training)
        x = self.dense_linear(x)
        x = self.dropout2(x, training=training)
        return x
