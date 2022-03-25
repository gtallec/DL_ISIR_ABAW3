import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import models.regressors.permutation_regressors.permutation_heuristics as order_heuristics

from models.layers.multi_order_attention_layers import MultiOrderDecoder, MOMTDecoder
from models.layers.dense import ParallelDense

class MultiOrderAttentionRegressor(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 M,
                 y_temperature=1.0,
                 xy_temperature=1.0,
                 rate=0.1,
                 use_encoder=True,
                 **kwargs):
        super(MultiOrderAttentionRegressor, self).__init__()
        order_sampler = order_heuristics.sample_with_heuristic({"type": "random"})

        # (P, T)
        orders = order_sampler(M, T)
        
        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = MultiOrderDecoder(num_layers=num_layers,
                                         d_model=d_model,
                                         num_heads=num_heads,
                                         dff=dff,
                                         T=T,
                                         orders=orders,
                                         y_temperature=y_temperature,
                                         xy_temperature=xy_temperature,
                                         rate=rate,
                                         use_encoder=use_encoder)
        self.orders = tf.Variable(orders,
                                  dtype=tf.int32,
                                  trainable=False)


        self.num_layers = num_layers
        self.final_layer = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')

        self.M = M
        self.T = T

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
   
        # (B, )
        order_indices = tf.squeeze(tf.random.categorical(tf.math.log(tf.ones((B, self.M)) /
                                                                     self.M),
                                                         num_samples=1),
                                   axis=-1)

        # (B, T)
        orders = tf.gather(self.orders, order_indices, axis=0)
        if training:
            # (B, T)
            y = 2 * y - 1
        else:
            # (B, T)
            y = tf.zeros((B, self.T))

        enc_output = self.encoding_compresser(inputs)
        enc_output = tf.reshape(enc_output, (B, -1, self.d_model))

        if training:
            # (B, T, d_model)
            dec_output_dict = self.decoder(x=y,
                                           enc_output=enc_output,
                                           training=training,
                                           order_indices=order_indices)
            # (B, T, 1)
            logits = self.final_layer(dec_output_dict['x'])
            # (B, T)
            logits = tf.squeeze(logits, axis=-1)
            output_dict['loss'] = logits
            output_dict['global_pred'] = tf.math.sigmoid(logits)
            for i in range(self.num_layers):
                output_dict['output_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block1'.format(i)]
                output_dict['inout_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block2'.format(i)]

        else:
            logits = tf.zeros((B, self.T))
            y_i = tf.zeros((B, ))
            fill_mask = tf.zeros((B, self.T))
            for i in range(self.T):
                # (B, T)
                y = y + y_i[:, tf.newaxis] * fill_mask_y

                # (B, T, d_model), (B, H, T, F)
                dec_output_dict = self.decoder(x=y,
                                               enc_output=enc_output,
                                               training=training,
                                               order_indices=order_indices)

                # (B, )
                task_i = orders[:, i]
                # (B, T, 1)
                logits_i = self.final_layer(dec_output_dict['x'])
                # (B, )
                logit_i = tf.gather(logits_i, task_i, axis=-1)
                # (B, )
                prob_i = tf.math.sigmoid(logit_i)
                sample_i = tf.random.uniform(shape=(B, ),
                                             minval=0,
                                             maxval=1.0)
                # (B, )
                y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1

                # #  Construct mask for filling y and logits at timestep i
                # (1, T)
                T_range = tf.range(0, self.T)[tf.newaxis, :]
                # (B, 1)
                task_i = orders[:, i][:, tf.newaxis]
                # (B, T)
                fill_mask_logit = tf.dtypes.cast(T_range == task_i, dtype=tf.float32)
                # (B, T + 1)
                fill_mask_y = tf.concat([tf.zeros((B, 1)), fill_mask_logit],
                                        axis=1)
                # (B, T)
                logits = logits + logit_i[:, tf.newaxis] * fill_mask_logit

            # (B, T)
            pred = tf.math.sigmoid(logits)
            output_dict['loss'] = logits
            output_dict['global_pred'] = pred
            for i in range(self.num_layers):
                output_dict['output_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block1'.format(i)]
                output_dict['inout_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block2'.format(i)]
            output_dict['token_dist_matrix'] = dec_output_dict['token_dist_matrix']
        return output_dict

class MOMTTransformer(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 M,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 use_encoder=True,
                 use_labels=True,
                 beam_search=False,
                 beam_width=0,
                 permutation_encoding=False,
                 shared_dense=True,
                 **kwargs):
        super(MOMTTransformer, self).__init__()
        order_sampler = order_heuristics.sample_with_heuristic({"type": "random"})

        # (M, T)
        orders = order_sampler(M, T)

        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = MOMTDecoder(num_layers=num_layers,
                                   d_model=d_model,
                                   num_heads=num_heads,
                                   dff=dff,
                                   T=T,
                                   orders=orders,
                                   temp_ty=temp_ty,
                                   temp_tx=temp_tx,
                                   rate=rate,
                                   use_encoder=use_encoder,
                                   use_labels=use_labels,
                                   permutation_encoding=permutation_encoding)

        self.orders = tf.Variable(orders,
                                  dtype=tf.int32,
                                  trainable=False)
        tf.print('self.orders : ', self.orders)

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
        self.beam_search = beam_search
        self.BW = beam_width

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
        # (B, 1, q_x, d)
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
            if not(self.beam_search):
                # (B, 1, T)
                orders = tf.gather(self.orders,
                                   indices=order_indices,
                                   axis=0)
                logits = tf.zeros((B, 1, self.T))
                y_i = tf.zeros((B, 1))
                # (1, T) -> (B, T)
                fill_mask_logit = tf.zeros((B, 1, self.T))
                for i in range(self.T):
                    # (B, 1, T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, 1, 1)), fill_mask_logit], axis=-1)
                    # (B, 1, T + 1)
                    y = y + y_i[:, tf.newaxis] * fill_mask_y


                    # (B, 1, T, d_model), (B, 1, H, T, F)
                    dec_output_dict = self.decoder(x=x,
                                                   y=y,
                                                   tokens=tokens,
                                                   order_indices=order_indices,
                                                   training=training)
                    # (B, 1)
                    task_i = orders[:, :, i]

                    # (B, 1, T)
                    logits_i = tf.squeeze(self.final_layer(dec_output_dict['x']), axis=-1)
                    # (B, 1)
                    logit_i = tf.gather(params=logits_i,
                                        indices=task_i,
                                        batch_dims=2,
                                        axis=-1)

                    # (B, 1)
                    prob_i = tf.math.sigmoid(logit_i)
                    sample_i = tf.random.uniform(shape=(B, 1),
                                                 minval=0,
                                                 maxval=1.0)

                    # (B, 1)
                    y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1

                    # (B, 1, T)
                    fill_mask_logit = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, 1, T)
                    logits = logits + logit_i[:, tf.newaxis] * fill_mask_logit

                # (B, 1, T)
                pred = tf.math.sigmoid(logits)
                pred = tf.math.reduce_mean(pred, axis=1)
                output_dict['loss'] = logits
                output_dict['global_pred'] = pred
                for i in range(self.num_layers):
                    output_dict['ty_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_ty'.format(i)]
                    output_dict['tx_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_tx'.format(i)]

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

class MOMTTWithController(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 M,
                 controllers,
                 temp_ty=1.0,
                 temp_tx=1.0,
                 rate=0.1,
                 use_labels=True,
                 beam_search=False,
                 beam_width=0,
                 permutation_encoding=False,
                 shared_dense=True,
                 **kwargs):
        super(MOMTTransformer, self).__init__()
        order_sampler = order_heuristics.sample_with_heuristic({"type": "random"})

        # (M, T)
        orders = order_sampler(M, T)

        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = MOMTDecoderWithController(num_layers=num_layers,
                                                 d_model=d_model,
                                                 num_heads=num_heads,
                                                 dff=dff,
                                                 encoder_controllers=encoder_controllers,
                                                 T=T,
                                                 orders=orders,
                                                 temp_ty=temp_ty,
                                                 temp_tx=temp_tx,
                                                 rate=rate,
                                                 use_labels=use_labels,
                                                 permutation_encoding=permutation_encoding)

        self.orders = tf.Variable(orders,
                                  dtype=tf.int32,
                                  trainable=False)
        tf.print('self.orders : ', self.orders)

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
        self.beam_search = beam_search
        self.BW = beam_width

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
        # (B, 1, q_x, d)
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
            if not(self.beam_search):
                # (B, 1, T)
                orders = tf.gather(self.orders,
                                   indices=order_indices,
                                   axis=0)
                logits = tf.zeros((B, 1, self.T))
                y_i = tf.zeros((B, 1))
                # (1, T) -> (B, T)
                fill_mask_logit = tf.zeros((B, 1, self.T))
                for i in range(self.T):
                    # (B, 1, T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, 1, 1)), fill_mask_logit], axis=-1)
                    # (B, 1, T + 1)
                    y = y + y_i[:, tf.newaxis] * fill_mask_y


                    # (B, 1, T, d_model), (B, 1, H, T, F)
                    dec_output_dict = self.decoder(x=x,
                                                   y=y,
                                                   tokens=tokens,
                                                   order_indices=order_indices,
                                                   training=training)
                    # (B, 1)
                    task_i = orders[:, :, i]

                    # (B, 1, T)
                    logits_i = tf.squeeze(self.final_layer(dec_output_dict['x']), axis=-1)
                    # (B, 1)
                    logit_i = tf.gather(params=logits_i,
                                        indices=task_i,
                                        batch_dims=2,
                                        axis=-1)

                    # (B, 1)
                    prob_i = tf.math.sigmoid(logit_i)
                    sample_i = tf.random.uniform(shape=(B, 1),
                                                 minval=0,
                                                 maxval=1.0)

                    # (B, 1)
                    y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1

                    # (B, 1, T)
                    fill_mask_logit = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, 1, T)
                    logits = logits + logit_i[:, tf.newaxis] * fill_mask_logit

                # (B, 1, T)
                pred = tf.math.sigmoid(logits)
                pred = tf.math.reduce_mean(pred, axis=1)
                output_dict['loss'] = logits
                output_dict['global_pred'] = pred
                for i in range(self.num_layers):
                    output_dict['ty_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_ty'.format(i)]
                    output_dict['tx_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_tx'.format(i)]

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


SUPPORTED_MULTI_ORDER_ATTENTION_REGRESSORS = {"moanet": MultiOrderAttentionRegressor,
                                              "momtt": MOMTTransformer}
