import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import models.regressors.permutation_regressors.permutation_heuristics as order_heuristics

from models.layers.block_attention_layers import BMOMTDecoder
from models.layers.dense import ParallelDense


class BMOMTTransformer(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 M,
                 controllers,
                 blocksize,
                 permutation_heuristic="random",
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 pred_mode='monte_carlo',
                 pred_N=10,
                 permutation_encoding=False,
                 shared_dense=True,
                 **kwargs):
        super(BMOMTTransformer, self).__init__()
        if permutation_heuristic == 'random': 
            order_sampler = order_heuristics.sample_with_heuristic({"type": permutation_heuristic})
        if permutation_heuristic == 'with_identity':
            order_sampler = order_heuristics.sample_with_heuristic({"type": "only_answer",
                                                                    "answer": "identity"})

        # (M, T)
        orders = order_sampler(M, T)

        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = BMOMTDecoder(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    controllers=controllers,
                                    T=T,
                                    orders=orders,
                                    blocksize=blocksize,
                                    temp_ty=temp_ty,
                                    temp_tx=temp_tx,
                                    rate=rate,
                                    permutation_encoding=permutation_encoding)

        self.orders = tf.Variable(orders,
                                  dtype=tf.int32,
                                  trainable=False)

        self.num_layers = num_layers
        if shared_dense:
            self.final_layer = tkl.Dense(units=1,
                                         activation='linear')
        else:
            self.final_layer = ParallelDense(units=1,
                                             T=T,
                                             activation='linear')
        self.T = T
        self.M = M
        self.blocksize = blocksize
        self.N_block = T // blocksize
        self.pred_mode = pred_mode
        self.pred_N = pred_N

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
        # (B, 1)
        order_indices = tf.random.categorical(tf.math.log(tf.ones((B, self.M)) /
                                                          self.M),
                                              num_samples=1)

        # (1, T, T) -> (B, T, T)
        tokens = tf.reshape(tf.eye(self.T), (1, 1, self.T, self.T)) 
        # Add start to groundtruth sequence.
        if training:
            # (B, T + 1)
            y = tf.concat([tf.zeros((B, 1)), 2 * y - 1], axis=1)
        else:
            # (B, T + 1)
            y = tf.zeros((B, self.T + 1))

        # (B, N_p, N_p, d)
        x = self.encoding_compresser(inputs)
        # (B, 1, q_x, d_model)
        x = tf.reshape(x, (B, 1, -1, self.d_model))
        # (B, 1, T + 1)
        y = tf.expand_dims(y, axis=-2)
 
        if training:
            # (B, 1, T, d_model)
            dec_output_dict = self.decoder(x=x,
                                           y=y,
                                           tokens=tokens,
                                           order_indices=order_indices,
                                           training=training)
            # (B, 1, T, 1)
            dec_output = dec_output_dict['x']
            logits = self.final_layer(dec_output)
            # (B, T)
            logits = tf.reshape(logits, (B, self.T))
            output_dict['loss'] = logits
            output_dict['global_pred'] = tf.math.sigmoid(logits)

        else:
            if self.pred_mode == 'monte_carlo':
                # For a given element sample N trajectories following the same order
                # (B, N)
                order_indices = tf.tile(order_indices, multiples=(1, self.pred_N))
                # (B, N, T)
                orders = tf.gather(self.orders,
                                   indices=order_indices,
                                   axis=0)

                # (B, N, q_x, d)
                x = tf.tile(x, multiples=(1, self.pred_N, 1, 1))
                # (B, N, N_block, blocksize)
                block_orders = tf.reshape(orders, (B, self.pred_N, self.N_block, self.blocksize))
                # (B, N, T)
                logits = tf.zeros((B, self.pred_N, self.T))
                # (B, N, blocksize)
                y_i = tf.zeros((B, self.pred_N, self.blocksize))
                # (B, N, blocksize, T)
                fill_mask_logit = tf.zeros((B, self.pred_N, self.blocksize, self.T))
                for i in range(self.N_block):
                    # (B, N, blocksize, T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, self.pred_N, self.blocksize, 1)),
                                             fill_mask_logit], axis=-1)
                    # (B, N, T + 1)
                    y = y + tf.math.reduce_sum(y_i[:, :, :, tf.newaxis] * fill_mask_y, axis=-2)

                    # (B, N, T, d_model), (B, N, H, T, F)
                    dec_output_dict = self.decoder(x=x,
                                                   y=y,
                                                   tokens=tokens,
                                                   order_indices=order_indices,
                                                   training=training)
                    # (B, N, blocksize)
                    block_i = block_orders[:, :, i, :]

                    # (B, N, T)
                    logits_i = tf.squeeze(self.final_layer(dec_output_dict['x']), axis=-1)
                    # (B, N, blocksize)
                    logit_i = tf.gather(params=logits_i,
                                        indices=block_i,
                                        batch_dims=2,
                                        axis=-1)

                    # (B, N, blocksize)
                    prob_i = tf.math.sigmoid(logit_i)
                    # (B, N, blocksize)
                    sample_i = tf.random.uniform(shape=(B, self.pred_N, self.blocksize),
                                                 minval=0,
                                                 maxval=1.0)

                    # (B, N, blocksize)
                    y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1

                    # (B, N, blocksize, T)
                    fill_mask_logit = tf.gather(tf.eye(self.T), block_i, axis=0)
                    # (B, N, T)
                    logits = logits + tf.math.reduce_sum(logit_i[:, :, :, tf.newaxis] * fill_mask_logit,
                                                         axis=-2)

                # (B, N, T)
                pred = tf.math.sigmoid(logits)
                output_dict['loss'] = tf.math.reduce_mean(logits, axis=1)
                output_dict['global_pred'] = tf.math.reduce_mean(pred, axis=1)
            else:
                # (B, 1, T, T)
                orders = tf.gather(self.orders,
                                   indices=order_indices,
                                   axis=0)
                # probability of N best sequence. (B, min(2^{i}, BW))
                log_p_Nbest = tf.zeros((B, 1))

                min_BW = 1

                for i in range(self.T):
                    # (B, min(2^{i}, BW), T, d_model), (B, min(2^{i}, BW), H, T, F)
                    x_tiled = tf.tile(x, (1, min_BW, 1, 1))
                    dec_output_dict = self.decoder(x=x_tiled,
                                                   y=y,
                                                   tokens=tokens,
                                                   order_indices=order_indices,
                                                   training=training)
                    # (B, 1)
                    task_i = orders[:, :, i]
                    # (B, 1, T)
                    fill_mask_task = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, 1, T + 1) -> (B, min(2^{i}, BW), T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, 1, 1)), fill_mask_task], axis=-1)

                    # (B, min(2^{i}, BW), T)
                    logits_i = tf.squeeze(self.final_layer(dec_output_dict['x']), axis=-1)
                    # (B, min(2^{i}, BW))
                    logit_i = tf.squeeze(tf.gather(params=logits_i,
                                                   indices=task_i,
                                                   batch_dims=1,
                                                   axis=-1),
                                         axis=-1)
                    ones_logits_i = tf.ones_like(logit_i)

                    # (B, min(2^{i}, BW))
                    log_p_i = -tf.math.softplus(-logit_i)
                    log_1_m_p_i = - tf.math.softplus(logit_i)


                    # (B, 2 * min(2^{i}, BW))
                    log_p_Nbest = (tf.tile(log_p_Nbest, (1, 2))
                                   + tf.concat([log_p_i, log_1_m_p_i], axis=-1))
                    # (B, 2 * min(2^{i}, BW), T + 1)
                    y = (tf.tile(y, (1, 2, 1))
                         + tf.concat([ones_logits_i, -ones_logits_i], axis=-1)[:, :, tf.newaxis]
                         * fill_mask_y)

                    # Take the BW most probable sequences:
                    min_BW = tf.math.minimum(2 ** (i + 1), self.BW)
                    # (B, min(2^{i + 1}, BW))
                    i_Nbest = tf.gather(tf.argsort(log_p_Nbest,
                                                   direction="DESCENDING",
                                                   axis=1),
                                        tf.range(0, min_BW), axis=1)
                    # (B, min(2^{i + 1}, BW))
                    log_p_Nbest = tf.gather(log_p_Nbest,
                                            i_Nbest,
                                            batch_dims=1,
                                            axis=1)

                    # (B, min(2^{i + 1}, BW), T + 1)
                    y = tf.gather(y,
                                  i_Nbest,
                                  batch_dims=1,
                                  axis=1)


                # (B, min(2^{i + 1}, BW))
                log_p_Nbest_normalized = log_p_Nbest - tf.math.reduce_logsumexp(log_p_Nbest, axis=-1)[:, tf.newaxis]

                pred = tf.math.reduce_sum(tf.math.maximum(tf.dtypes.cast(0, dtype=tf.float32), y)
                                          * tf.math.exp(log_p_Nbest_normalized)[:, :, tf.newaxis],
                                          axis=1)[:, 1:]
                output_dict['global_pred'] = tf.clip_by_value(pred, 0, 1)
                output_dict['loss'] = tf.math.log(pred / (1 - pred))
        return output_dict


SUPPORTED_BLOCK_ATTENTION_REGRESSORS = {"bmomtt": BMOMTTransformer}
