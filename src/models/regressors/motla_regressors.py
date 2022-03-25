import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import models.regressors.permutation_regressors.permutation_heuristics as order_heuristics

from models.layers.motla_layers import MOTLADecoder
from models.layers.dense import ParallelDense


class MOTLAT(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 M,
                 controllers,
                 permutation_heuristic="with_identity",
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 pred_mode='monte_carlo',
                 pred_N=10,
                 permutation_encoding=False,
                 shared_dense=True,
                 **kwargs):
        super(MOTLAT, self).__init__()
        if permutation_heuristic == 'random': 
            order_sampler = order_heuristics.sample_with_heuristic({"type": permutation_heuristic})
        if permutation_heuristic == 'with_identity':
            order_sampler = order_heuristics.sample_with_heuristic({"type": "with_answer",
                                                                    "answer": "identity"})

        # (M, T)
        orders = order_sampler(M, T)

        self.num_heads = num_heads
        self.d_model = d_model

        self.decoder = MOTLADecoder(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    controllers=controllers,
                                    T=T,
                                    orders=orders,
                                    temp_ty=temp_ty,
                                    temp_tx=temp_tx,
                                    rate=rate,
                                    permutation_encoding=permutation_encoding)

        self.orders = tf.Variable(orders,
                                  dtype=tf.int32,
                                  trainable=False)

        self.num_layers = num_layers
        if shared_dense:
            self.final_dense = tkl.Dense(units=1,
                                         activation='linear')
        else:
            self.final_dense = ParallelDense(units=1,
                                             T=T,
                                             activation='linear')

        self.T = T
        self.M = M
        self.pred_mode = pred_mode
        self.pred_N = pred_N

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
        # (B, 1)
        order_indices = tf.random.categorical(tf.math.log(tf.ones((B, self.M)) /
                                                          self.M),
                                              num_samples=1)

        # (T, T) -> (1, 1, T, T)
        tokens = tf.reshape(tf.eye(self.T), (1, 1, self.T, self.T)) 
        # Add start to groundtruth sequence.
        if training:
            # (B, T + 1)
            y = tf.concat([tf.zeros((B, 1)), 2 * y - 1], axis=1)
        else:
            # (B, T + 1)
            y = tf.zeros((B, self.T + 1))

        # (B, 1, N_patch, d_model)
        x = tf.expand_dims(inputs, axis=-2)
        # (B, 1, T + 1)
        y = tf.expand_dims(y, axis=-2)
 
        if training:
            # (B, 1, T, d_model)
            dec_output_dict = self.decoder(x=x,
                                           y=y,
                                           tokens=tokens,
                                           order_indices=order_indices,
                                           training=training)
            # (B, 1, T, d_model)
            dec_output = dec_output_dict['x']
            logits = tf.squeeze(self.final_dense(dec_output), axis=-1)
            # (B, 1, T)
            # (B, T)
            logits = tf.reshape(logits, (B, self.T))
            output_dict['loss'] = logits
            output_dict['global_pred'] = tf.math.sigmoid(logits)

            for i in range(self.num_layers):
                output_dict['ty_att_{}'.format(i)] = tf.squeeze(dec_output_dict['dec_layer{}_block_ty'.format(i)],
                                                                axis=1)
                output_dict['tx_att_{}'.format(i)] = tf.squeeze(dec_output_dict['dec_layer{}_block_tx'.format(i)],
                                                                axis=1)

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
                # (B, N, T)
                logits = tf.zeros((B, self.pred_N, self.T))
                # (B, N, 1)
                y_i = tf.zeros((B, self.pred_N))
                # (B, N, T)
                fill_mask_logit = tf.zeros((B, self.pred_N, self.T))
                for i in range(self.T):
                    # (B, N, T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, self.pred_N, 1)), fill_mask_logit], axis=-1)
                    # (B, N, T + 1)
                    y = y + y_i[:, :, tf.newaxis] * fill_mask_y

                    # (B, N, T, d_model), (B, N, H, T, F)
                    dec_output_dict = self.decoder(x=x,
                                                   y=y,
                                                   tokens=tokens,
                                                   order_indices=order_indices,
                                                   training=training)
                    # (B, N)
                    task_i = orders[:, :, i]

                    # (B, N, T)
                    logits_i = tf.squeeze(self.final_dense(dec_output_dict['x']), axis=-1)
                    # (B, N, 1)
                    logit_i = tf.gather(params=logits_i,
                                        indices=task_i,
                                        batch_dims=2,
                                        axis=-1)
                    # (B, N)
                    prob_i = tf.math.sigmoid(logit_i)
                    sample_i = tf.random.uniform(shape=(B, self.pred_N),
                                                 minval=0,
                                                 maxval=1.0)

                    # (B, N)
                    y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1

                    # (B, N, T)
                    fill_mask_logit = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, N, T)
                    logits = logits + logit_i[:, :, tf.newaxis] * fill_mask_logit

                # (B, N, T)
                pred = tf.math.sigmoid(logits)
                output_dict['loss'] = tf.math.reduce_mean(logits, axis=1)
                output_dict['global_pred'] = tf.math.reduce_mean(pred, axis=1)
                for i in range(self.num_layers):
                    output_dict['ty_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_ty'.format(i)]
                    output_dict['tx_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_tx'.format(i)]

            elif self.pred_mode == 'beam_search':
                # (B, 1, T, T)
                orders = tf.gather(self.orders,
                                   indices=order_indices,
                                   axis=0)
                # probability of N best sequence. (B, min(2^{i}, pred_N))
                log_p_Nbest = tf.zeros((B, 1))

                min_pred_N = 1

                for i in range(self.T):
                    # (B, min(2^{i}, pred_N), T, d_model), (B, min(2^{i}, pred_N), H, T, F)
                    x_tiled = tf.tile(x, (1, min_pred_N, 1, 1))
                    dec_output_dict = self.decoder(x=x_tiled,
                                                   y=y,
                                                   tokens=tokens,
                                                   order_indices=order_indices,
                                                   training=training)
                    # (B, 1)
                    task_i = orders[:, :, i]
                    # (B, 1, T)
                    fill_mask_task = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, 1, T + 1) -> (B, min(2^{i}, pred_N), T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, 1, 1)), fill_mask_task], axis=-1)

                    # (B, min(2^{i}, pred_N), T)
                    logits_i = tf.squeeze(self.final_dense(dec_output_dict['x']), axis=-1)
                    # (B, min(2^{i}, pred_N))
                    logit_i = tf.squeeze(tf.gather(params=logits_i,
                                                   indices=task_i,
                                                   batch_dims=1,
                                                   axis=-1),
                                         axis=-1)
                    ones_logits_i = tf.ones_like(logit_i)

                    # (B, min(2^{i}, pred_N))
                    log_p_i = -tf.math.softplus(-logit_i)
                    log_1_m_p_i = - tf.math.softplus(logit_i)


                    # (B, 2 * min(2^{i}, pred_N))
                    log_p_Nbest = (tf.tile(log_p_Nbest, (1, 2))
                                   + tf.concat([log_p_i, log_1_m_p_i], axis=-1))
                    # (B, 2 * min(2^{i}, pred_N), T + 1)
                    y = (tf.tile(y, (1, 2, 1))
                         + tf.concat([ones_logits_i, -ones_logits_i], axis=-1)[:, :, tf.newaxis]
                         * fill_mask_y)

                    # Take the pred_N most probable sequences:
                    min_pred_N = tf.math.minimum(2 ** (i + 1), self.pred_N)
                    # (B, min(2^{i + 1}, pred_N))
                    i_Nbest = tf.gather(tf.argsort(log_p_Nbest,
                                                   direction="DESCENDING",
                                                   axis=1),
                                        tf.range(0, min_pred_N), axis=1)
                    # (B, min(2^{i + 1}, pred_N))
                    log_p_Nbest = tf.gather(log_p_Nbest,
                                            i_Nbest,
                                            batch_dims=1,
                                            axis=1)

                    # (B, min(2^{i + 1}, pred_N), T + 1)
                    y = tf.gather(y,
                                  i_Nbest,
                                  batch_dims=1,
                                  axis=1)


                # (B, min(2^{i + 1}, pred_N))
                log_p_Nbest_normalized = log_p_Nbest - tf.math.reduce_logsumexp(log_p_Nbest, axis=-1)[:, tf.newaxis]

                pred = tf.math.reduce_sum(tf.math.maximum(tf.dtypes.cast(0, dtype=tf.float32), y)
                                          * tf.math.exp(log_p_Nbest_normalized)[:, :, tf.newaxis],
                                          axis=1)[:, 1:]
                output_dict['global_pred'] = tf.clip_by_value(pred, 0, 1)
                output_dict['loss'] = tf.math.log(pred / (1 - pred))
        return output_dict

class TLAT(MOTLAT):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 controllers,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 pred_mode='beam_search',
                 pred_N=10,
                 shared_dense=True,
                 **kwargs): 
        super(TLAT, self).__init__(num_layers=num_layers,
                                   d_model=d_model,
                                   num_heads=num_heads,
                                   dff=dff,
                                   T=T,
                                   M=1,
                                   controllers=controllers,
                                   permutation_heuristic="with_identity",
                                   temp_ty=temp_ty,
                                   temp_tx=temp_tx,
                                   rate=rate,
                                   pred_mode=pred_mode,
                                   pred_N=pred_N,
                                   permutation_encoding=False,
                                   shared_dense=shared_dense,
                                   **kwargs)




