import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

from models.layers.lla_layers import LLADecoder
from models.layers.dense import ParallelDense

def create_look_ahead_mask(size):
    # (T, T)
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

# Label to Label Attention transformer
class LLAT(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 attention_mode,
                 controllers,
                 temp_yy=1.0,
                 temp_yx=1.0,
                 rate=0.1,
                 pred_mode='monte_carlo',
                 pred_N=10,
                 shared_dense=True,
                 **kwargs):
        super(LLAT, self).__init__()
        self.num_heads = num_heads

        self.d_model = d_model
        self.decoder = LLADecoder(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  attention_mode=attention_mode,
                                  controllers=controllers,
                                  T=T,
                                  temp_yy=temp_yy,
                                  temp_yx=temp_yx,
                                  rate=rate)

        self.num_layers = num_layers
        if shared_dense:
            self.final_dense = tkl.Dense(units=1,
                                         activation='linear')
        else:
            self.final_dense = ParallelDense(units=1,
                                             T=T,
                                             activation='linear')

        self.T = T
        self.pred_mode = pred_mode
        self.pred_N = pred_N
        self.look_ahead_mask = create_look_ahead_mask(self.T)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    def call(self, inputs, y=None, training=None):
        """ inputs of size (B, N_patch, d_model) """
        output_dict = dict()
        B = tf.shape(inputs)[0]
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
            look_ahead_mask = tf.tile(self.look_ahead_mask, (B, 1, 1, 1, 1))
            # (B, 1, T, d_model)
            dec_output_dict = self.decoder(x=x,
                                           y=y,
                                           look_ahead_mask=look_ahead_mask,
                                           training=training)
            # (B, 1, T, d_model)
            dec_output = dec_output_dict['x']
            logits = tf.squeeze(self.final_dense(dec_output), axis=-1)
            # (B, T)
            logits = tf.reshape(logits, (B, self.T))
            output_dict['loss'] = logits
            output_dict['global_pred'] = tf.math.sigmoid(logits)

            for i in range(self.num_layers):
                output_dict['yy_att_{}'.format(i)] = tf.squeeze(dec_output_dict['dec_layer{}_block_yy'.format(i)],
                                                                axis=1)
                output_dict['yx_att_{}'.format(i)] = tf.squeeze(dec_output_dict['dec_layer{}_block_yx'.format(i)],
                                                                axis=1)

        else:
            look_ahead_mask = tf.tile(self.look_ahead_mask, (B, self.pred_N, 1, 1, 1))
            # (B, N, T)
            orders = tf.tile(tf.range(self.T)[tf.newaxis, tf.newaxis, :],
                             (B, self.pred_N, 1))
            if self.pred_mode == 'monte_carlo':
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
                                                   look_ahead_mask=look_ahead_mask,
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
                    output_dict['yy_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_yy'.format(i)]
                    output_dict['yx_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_yx'.format(i)]

            elif self.pred_mode == 'beam_search':
                # probability of N best sequence. (B, min(2^{i}, pred_N))
                log_p_Nbest = tf.zeros((B, 1))
                min_pred_N = 1
                for i in range(self.T):
                    # (B, min(2^{i}, pred_N), T, d_model), (B, min(2^{i}, pred_N), H, T, F)
                    x_tiled = tf.tile(x, (1, min_pred_N, 1, 1))
                    dec_output_dict = self.decoder(x=x_tiled,
                                                   y=y,
                                                   look_ahead_mask=look_ahead_mask,
                                                   training=training)
                    # (B, 1)
                    task_i = orders[:, :, i]
                    # (B, 1, T)
                    fill_mask_task = tf.gather(tf.eye(self.T), task_i, axis=0)
                    # (B, 1, T + 1) -> (B, min(2^{i}, pred_N), T + 1)
                    fill_mask_y = tf.concat([tf.zeros((B, 1, 1)), fill_mask_task], axis=-1)

                    # (B, min(2^{i}, pred_N), T)
                    logits_i = tf.squeeze(self.final_kernel(dec_output_dict['x']), axis=-1)
                    logits_i = logits_i + self.final_bias
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
                    
                    # Compute log_proba of sequence
                    # (B, 2 * min(2^{i}, pred_N))
                    log_p_Nbest = (tf.tile(log_p_Nbest, (1, 2))
                                   + tf.concat([log_p_i, log_1_m_p_i], axis=-1))

                    # Update associated sequences
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


SUPPORTED_CONTROLLED_ATTENTION_REGRESSORS = {"lla": LLAT}
