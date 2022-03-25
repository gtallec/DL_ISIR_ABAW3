import numpy as np
import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import models.regressors.permutation_regressors.permutation_heuristics as order_heuristics

from models.layers.motcat_layers import TMCALayers, LSALayers, MultiLabelEncoder

def generate_look_ahead_masks(orders):
    """ Generate the look ahead mask for M different orders of T tasks
    Input : orders of size (M, T)
    Output : look ahead mask of size (M, T, T)
    """
    M, T = orders.shape
    look_ahead_masks = np.zeros((M, T, T))
    for m in range(M):
        for t in range(T):
            task = orders[m, t]
            look_ahead_masks[m, task, task] = 1
            for previous_t in range(t):
                previous_task = orders[m, previous_t]
                look_ahead_masks[m, task, previous_task] = 1
    return look_ahead_masks
                


class MOTCAT(tkm.Model):
    def __init__(self,
                 # General Architecture
                 T,
                 # Token Branch
                 num_layers_t,
                 num_layers_y,
                 d_model_t,
                 d_model_y,
                 mlp_scale_t,
                 mlp_scale_y,
                 num_heads_t,
                 num_heads_y,
                 rate_t,
                 rate_y,
                 rate_corrupt,
                 temp_tx,
                 temp_ty,
                 temp_yy,
                 # Permutation Concerns
                 M,
                 permutation_heuristic="random",
                 permutation_encoding=False,
                 start_token_attention=True,
                 # Inference
                 pred_mode='monte_carlo',
                 ca_order="xy",
                 pred_N=10,
                 **kwargs):
        super(MOTCAT, self).__init__()
        dff_t = d_model_t * mlp_scale_t
        dff_y = d_model_y * mlp_scale_y
        # Orders concerns
        if permutation_heuristic == 'random':
            order_sampler = order_heuristics.sample_with_heuristic({"type": permutation_heuristic})
        if permutation_heuristic == 'with_identity':
            order_sampler = order_heuristics.sample_with_heuristic({"type": "with_answer",
                                                                    "answer": "identity"})
        # (M, T)
        orders = order_sampler(M, T)
        # (M, T, T)
        self.look_ahead_masks = tf.Variable(generate_look_ahead_masks(orders),
                                            dtype=tf.float32,
                                            trainable=False)

        self.orders = tf.Variable(orders,
                                  dtype=tf.int32,
                                  trainable=False)
        # Tokens and Labels concerns
        embedding_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        # Tokens Concerns
        self.token_embedder = tkl.Dense(units=d_model_t,
                                        activation='linear',
                                        kernel_initializer=embedding_init,
                                        use_bias=True,
                                        name='token_embedder')

        self.tmca_layers = TMCALayers(num_layers=num_layers_t,
                                      d_model=d_model_t,
                                      num_heads=num_heads_t,
                                      ca_order=ca_order,
                                      dff=dff_t,
                                      temp_tx=temp_tx,
                                      temp_ty=temp_ty,
                                      rate=rate_t)
        self.multi_label_encoder = MultiLabelEncoder(d_model_y, T)
        self.lsa_layers = LSALayers(num_layers=num_layers_y,
                                    d_model=d_model_y,
                                    num_heads=num_heads_y,
                                    dff=dff_y,
                                    temp=temp_yy,
                                    rate=rate_y)

        self.final_dense = tkl.Dense(units=1,
                                     activation='linear',
                                     name='pred_dense')

        self.T = T
        self.ca_order = ca_order
        self.M = M
        self.pred_mode = pred_mode
        self.pred_N = pred_N
        self.d_model_t = d_model_t
        self.d_model_y = d_model_y
        self.start_token_attention = start_token_attention
        self.rate_corrupt = rate_corrupt

    def call(self, x, y=None, training=None):
        """
        x (B, N_patch, d_model)
        y (B, T)
        """
        output_dict = dict()
        shape_x = tf.shape(x)
        B = shape_x[0]
        N_patch = shape_x[1]
        # Label corruption in train for better train/test conditional distrib similarity
        if training:
            # (B, T)
            label_noise = tf.random.uniform(shape=(B, self.T),
                                            minval=0,
                                            maxval=1,
                                            dtype=tf.float32)
            # (B, T)
            # 1 if label is corrupted 0 else
            label_corrupt = tf.dtypes.cast(label_noise <= self.rate_corrupt, tf.float32)
            y = ((2 * y - 1) * (1 - 2 * label_corrupt) + 1)/2

        # (B, 1)
        order_indices = tf.random.categorical(tf.math.log(tf.ones((B, self.M)) /
                                                          self.M),
                                              num_samples=1)
        # (B, 1, T, T)
        mask_y = tf.gather(self.look_ahead_masks,
                           order_indices,
                           axis=0)
        # (B, 1, 1, T, T) -> (B, N, H, T, T)
        mask_y_sa = tf.expand_dims(mask_y,
                                   axis=-3)
        # mask_y_sa = tf.ones_like(mask_y_sa)

        if not(self.start_token_attention):
            # (B, T)
            batch_orders = tf.squeeze(tf.gather(self.orders, order_indices, axis=0), axis=1)
            # (B, T)
            start_attention_col = tf.gather(tf.eye(self.T), batch_orders[:, 0], axis=0)
            start_attention_col = tf.reshape(start_attention_col, (B, 1, 1, self.T, 1))
        else:
            start_attention_col = tf.ones((B, 1, 1, self.T, 1))

        # (B, 1, 1, T, T + 1) -> (B, N, H, T, T + 1)
        mask_y_ca = tf.concat([start_attention_col,
                               mask_y_sa - tf.reshape(tf.eye(self.T), (1, 1, 1, self.T, self.T))],
                              axis=-1)

        # mask_y_ca = tf.ones_like(mask_y_ca)


        # (1, 1, 1, T, N_patch) -> (B, 1, 1, T, N_patch)
        mask_x = tf.ones((1, 1, 1, self.T, N_patch))

        # (1, 1, T, T) -> (B, N, T, T)
        tokens = tf.reshape(tf.eye(self.T), (1, 1, self.T, self.T))
        # (1, 1, T, d_model)
        tokens = self.token_embedder(tokens)

        if training:
            # (B, T)
            y = 2 * y - 1
        else:
            # (B, T)
            y = tf.zeros((B, self.T))

        # (B, 1, N_patch, d_model_x)
        x = tf.expand_dims(x, axis=-3)
        # (B, 1, T)
        y = tf.expand_dims(y, axis=-2)
        # (B, 1, 1, d_model)
        start_y = tf.zeros((1, 1, 1, self.d_model_y))

        if training:
            # (B, 1, T, d_model_y)
            y_enc = self.multi_label_encoder(y)
            # (B, 1, T, d_model_y)
            tokens = tf.tile(tokens, (B, 1, 1, 1))
            start_y = tf.tile(start_y, (B, 1, 1, 1))

            # (B, 1, T, T)
            y_enc, blocks_yy = self.lsa_layers(y=y_enc,
                                               mask=mask_y_sa,
                                               training=training)
            # (B, 1, T + 1, d_model_y)
            y_with_start = tf.concat([start_y, y_enc], axis=-2)

            # (B, 1, T, d_model_t)
            prelogits, blocks = self.tmca_layers(tokens=tokens,
                                                 x=x,
                                                 y=y_with_start,
                                                 mask_x=mask_x,
                                                 mask_y=mask_y_ca,
                                                 training=training)
            # tf.print('block_ty : ', tf.math.reduce_mean(blocks[0][0], axis=1)[0])
            # (B, 1, T, 1)
            logits = tf.reshape(self.final_dense(prelogits), (B, self.T))

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
                # (B, N, 1, d_model)

                tokens = tf.tile(tokens, (B, self.pred_N, 1, 1))
                start_y = tf.tile(start_y, (B, self.pred_N, 1, 1))
                # (B, N, T)
                logits = tf.zeros((B, self.pred_N, self.T))
                # (B, N, 1)
                y_i = tf.zeros((B, self.pred_N))
                # (B, N, T)
                fill_mask = tf.zeros((B, self.pred_N, self.T))
                for i in range(self.T):
                    # (B, N, T)
                    y = y + y_i[:, :, tf.newaxis] * fill_mask
                    # (B, N, T, d_model_y)
                    y_enc = self.multi_label_encoder(y)
                    # (B, N, T, d_model_y)
                    y_enc, blocks_yy = self.lsa_layers(y=y_enc,
                                                       mask=mask_y_sa,
                                                       training=training)
                    # (B, N, T + 1, d_model_y)
                    y_enc = tf.concat([start_y, y_enc], axis=-2)
                    # (B, N, T, d_model_t), (B, N, H, T, F)
                    prelogits, blocks = self.tmca_layers(tokens=tokens,
                                                         x=x,
                                                         y=y_enc,
                                                         mask_x=mask_x,
                                                         mask_y=mask_y_ca,
                                                         training=training)

                    # (B, N, T)
                    logits_i = tf.squeeze(self.final_dense(prelogits), axis=-1)
                    # (B, N)
                    task_i = orders[:, :, i]

                    # (B, N)
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
                    fill_mask = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, N, T)
                    logits = logits + tf.expand_dims(logit_i, axis=-1) * fill_mask

                # (B, N, T)
                pred = tf.math.sigmoid(logits)
                output_dict['loss'] = tf.math.reduce_mean(logits, axis=1)
                output_dict['global_pred'] = tf.math.reduce_mean(pred, axis=1)

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

                    y_enc = self.label_encoder(y=y,
                                               order_indices=order_indices,
                                               training=training)

                    dec_output_dict = self.decoder(x=x_tiled,
                                                   y=y_enc,
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
        for i in range(len(blocks)):
            if self.ca_order == 'xy':
                output_dict['layer_{}_block_tx'.format(i)] = tf.math.reduce_mean(blocks[i][0],
                                                                                 axis=1)
                output_dict['layer_{}_block_ty'.format(i)] = tf.math.reduce_mean(blocks[i][1],
                                                                                 axis=1)

            elif self.ca_order == 'yx':
                output_dict['layer_{}_block_ty'.format(i)] = tf.math.reduce_mean(blocks[i][0],
                                                                                 axis=1)
                output_dict['layer_{}_block_tx'.format(i)] = tf.math.reduce_mean(blocks[i][1],
                                                                                 axis=1)
        return output_dict

class TCAT(MOTCAT):
    def __init__(self,
                 # General Architecture
                 T,
                 shared_dense,
                 # Token Branch
                 num_layers_t,
                 d_model_t,
                 dff_t,
                 num_heads_t,
                 rate_t,
                 temp_tx,
                 temp_ty,
                 # Label Branch
                 num_layers_y,
                 d_model_y,
                 dff_y,
                 num_heads_y,
                 rate_y,
                 temp_yy,
                 # Permutation Concerns
                 M,
                 permutation_heuristic="random",
                 permutation_encoding=False,
                 # Inference
                 pred_mode='monte_carlo',
                 pred_N=10,
                 **kwargs): 
        super(MOTCAT, self).__init__(self,
                                     T,
                                     shared_dense,
                                     # Token Branch
                                     num_layers_t,
                                     d_model_t,
                                     dff_t,
                                     num_heads_t,
                                     rate_t,
                                     temp_tx,
                                     temp_ty,
                                     # Label Branch
                                     num_layers_y,
                                     d_model_y,
                                     dff_y,
                                     num_heads_y,
                                     rate_y,
                                     temp_yy,
                                     # Permutation Concerns
                                     M=1,
                                     permutation_heuristic="with_identity",
                                     permutation_encoding=False,
                                     # Inference
                                     pred_mode='monte_carlo',
                                     pred_N=10,
                                     **kwargs)




