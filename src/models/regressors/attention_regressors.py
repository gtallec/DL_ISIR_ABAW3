import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
from models.layers.attention_layers import Encoder, Decoder, UncertaintyDecoder, MultiTaskDecoder
from models.layers.dense import ParallelDense



def create_padding_mask(seq):
    seq_mask = tf.zeros_like(seq)
    return seq_mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

class AttentionRegressor(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 label_temperature=1.0,
                 rate=0.1,
                 use_encoder=True,
                 mask_start_token=False,
                 **kwargs):
        super(AttentionRegressor, self).__init__()
        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               T=T,
                               label_temperature=label_temperature,
                               rate=rate,
                               use_encoder=use_encoder)
        self.num_layers = num_layers
        self.final_layer = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')
        self.T = T
        self.mask_start_token = mask_start_token

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
        # Shift target sequence right.
        if training:
            # (B, T)
            y = tf.concat([tf.zeros((B, 1)), (2 * y - 1)[:, :-1]], axis=1)
        else:
            # (B, T)
            y = tf.zeros((B, self.T))

        enc_output = self.encoding_compresser(inputs)
        enc_output = tf.reshape(enc_output, (B, -1, self.d_model))
        n_q = tf.shape(enc_output)[1]

        
        # (1, 1, T, T) -> (B, H, T, T)
        look_ahead_mask = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0)[tf.newaxis, tf.newaxis, :, :]

        if self.mask_start_token:
            start_token_mask = tf.concat([[1], tf.zeros((self.T - 1, ))], axis=0)[:, tf.newaxis]
            start_token_mask = tf.concat([start_token_mask, tf.ones((self.T, self.T - 1))], axis=1)
            look_ahead_mask = look_ahead_mask * start_token_mask

 
        # (1, 1, T, ) -> (B, H, T, d_model)
        dec_padding_mask = tf.ones((1, 1, self.T, n_q))

        if training:
            # (B, T, d_model)
            dec_output_dict = self.decoder(x=y,
                                           enc_output=enc_output,
                                           training=training,
                                           look_ahead_mask=look_ahead_mask,
                                           padding_mask=dec_padding_mask)
            # (B, T, 1)
            logits = self.final_layer(dec_output_dict['x'])
            # (B, T)
            logits = tf.squeeze(logits, axis=-1)
            output_dict['loss'] = logits
            output_dict['global_pred'] = tf.math.sigmoid(logits)
            for i in range(self.num_layers):
                output_dict['output_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block1'.format(i)]
                output_dict['inout_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block2'.format(i)]
            output_dict['token_dist_matrix'] = dec_output_dict['token_dist_matrix']

        else:
            logits = tf.zeros((B, self.T))
            y_i = tf.zeros((B, ))
            for i in range(self.T):
                # (B, T)
                fill_mask = tf.concat([tf.zeros((B, i)),
                                       tf.ones((B, 1)),
                                       tf.zeros((B, self.T - (i + 1)))],
                                      axis=1)
                y = y + y_i[:, tf.newaxis] * fill_mask

                # (B, T, d_model), (B, H, T, F)
                dec_output_dict = self.decoder(x=y,
                                               enc_output=enc_output,
                                               training=training,
                                               look_ahead_mask=look_ahead_mask,
                                               padding_mask=dec_padding_mask)

                # (B, T, 1)
                logits_i = self.final_layer(dec_output_dict['x'])
                # (B, )
                logit_i = tf.squeeze(logits_i, axis=-1)[:, i]
                # (B, )
                prob_i = tf.math.sigmoid(logit_i)
                sample_i = tf.random.uniform(shape=(B, ),
                                             minval=0,
                                             maxval=1.0)

                # (B, )
                y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1
                # (B, T)
                logits = logits + logit_i[:, tf.newaxis] * fill_mask

            # (B, T)
            pred = tf.math.sigmoid(logits)
            output_dict['loss'] = logits
            output_dict['global_pred'] = pred
            for i in range(self.num_layers):
                output_dict['output_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block1'.format(i)]
                output_dict['inout_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block2'.format(i)]
            output_dict['token_dist_matrix'] = dec_output_dict['token_dist_matrix']
        return output_dict

class UAttentionRegressor(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 rate=0.1, 
                 **kwargs):
        super(UAttentionRegressor, self).__init__()
        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               T=T,
                               rate=rate)

        self.final_layer = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')

        self.T = T
        self.log_u = tf.Variable(tf.zeros((self.T, )), trainable=True)

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
        inv_u2 = tf.math.exp(-2 * self.log_u)
        # Shift target sequence right.
        if training:
            # (B, T)
            y = tf.concat([tf.zeros((B, 1)), (2 * y - 1)[:, :-1]], axis=1)
        else:
            # (B, T)
            y = tf.zeros((B, self.T))

        F = tf.shape(inputs)[-1]
        enc_output = tf.reshape(inputs, (B, -1, F))
        enc_output = self.encoding_compresser(enc_output)

        
        # (1, 1, T, T) -> (B, H, T, T)
        look_ahead_mask = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0)[tf.newaxis, tf.newaxis, :, :]
 
        # (1, 1, T, d_model) -> (B, H, T, d_model)
        dec_padding_mask = tf.ones((1, 1, self.T, self.d_model))

        if training:
            # (B, T, d_model)
            dec_output_dict = self.decoder(x=y,
                                           enc_output=enc_output,
                                           training=training,
                                           look_ahead_mask=look_ahead_mask,
                                           padding_mask=dec_padding_mask)
            # (B, T, 1)
            logits = self.final_layer(dec_output_dict['x'])
            # (B, T)
            logits = tf.squeeze(logits, axis=-1)
            output_dict['logits'] = logits
            output_dict['log_u'] = self.log_u[tf.newaxis, :]
            output_dict['global_pred'] = tf.math.sigmoid(logits * inv_u2)
            output_dict.update(dec_output_dict)

        else:
            logits = tf.zeros((B, self.T))
            y_i = tf.zeros((B, ))
            for i in range(self.T):
                # (B, T)
                fill_mask = tf.concat([tf.zeros((B, i)),
                                       tf.ones((B, 1)),
                                       tf.zeros((B, self.T - (i + 1)))],
                                      axis=1)
                y = y + y_i[:, tf.newaxis] * fill_mask

                # (B, T, d_model), (B, H, T, F)
                dec_output_dict = self.decoder(x=y,
                                               enc_output=enc_output,
                                               training=training,
                                               look_ahead_mask=look_ahead_mask,
                                               padding_mask=dec_padding_mask)

                # (B, T, 1)
                logits_i = self.final_layer(dec_output_dict['x'])
                # (B, )
                logit_i = tf.squeeze(logits_i, axis=-1)[:, i]
                # (B, )
                prob_i = tf.math.sigmoid(logit_i * inv_u2[i])
                sample_i = tf.random.uniform(shape=(B, ),
                                             minval=0,
                                             maxval=1.0)

                # (B, )
                y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1
                # (B, T)
                logits = logits + logit_i[:, tf.newaxis] * fill_mask

            # (B, T)
            pred = tf.math.sigmoid(logits * inv_u2[tf.newaxis, :])
            output_dict['log_u'] = self.log_u[tf.newaxis, :]
            output_dict['logits'] = logits
            output_dict['global_pred'] = pred
        return output_dict



class RUAttentionRegressor(tkm.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 T,
                 rate=0.1, 
                 **kwargs):
        super(RUAttentionRegressor, self).__init__()
        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               T=T,
                               rate=rate)
        self.final_layer = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')

        self.uncertainty_decoder = UncertaintyDecoder(num_layers=num_layers,
                                                      d_model=d_model,
                                                      num_heads=num_heads,
                                                      dff=dff,
                                                      T=T,
                                                      rate=rate)
        self.uncertainty_dense = ParallelDense(units=1,
                                               T=T,
                                               kernel_initializer='zeros',
                                               activation='linear')
        self.T = T

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
        # Shift target sequence right.
        if training:
            # (B, T)
            y = tf.concat([tf.zeros((B, 1)), (2 * y - 1)[:, :-1]], axis=1)
        else:
            # (B, T)
            y = tf.zeros((B, self.T))

        F = tf.shape(inputs)[-1]
        enc_output = tf.reshape(inputs, (B, -1, F))
        enc_output = self.encoding_compresser(enc_output)

        
        # (1, 1, T, T) -> (B, H, T, T)
        look_ahead_mask = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0)[tf.newaxis, tf.newaxis, :, :]
 
        # (1, 1, T, d_model) -> (B, H, T, d_model)
        dec_padding_mask = tf.ones((1, 1, self.T, self.d_model))

        if training:
            # (B, T, d_model)
            dec_output_dict = self.decoder(x=y,
                                           enc_output=enc_output,
                                           training=training,
                                           look_ahead_mask=look_ahead_mask,
                                           padding_mask=dec_padding_mask)
            # (B, T, d_model)
            uncertainty_output = self.uncertainty_decoder(x=y,
                                                          training=training,
                                                          look_ahead_mask=look_ahead_mask)

            # (B, T, 1)
            logits = self.final_layer(dec_output_dict['x'])
            # (B, T)
            logits = tf.squeeze(logits, axis=-1)

            # (B, T, 1)
            log_u = self.uncertainty_dense(uncertainty_output)
            # (B, T)
            log_u = tf.squeeze(log_u, axis=-1)
            # (B, T)
            inv_u2 = tf.math.exp(-2 * log_u)

            # (B, T)
            output_dict['logits'] = logits
            output_dict['log_u'] = log_u
            output_dict['global_pred'] = tf.math.sigmoid(logits * inv_u2)
            output_dict.update(dec_output_dict)

        else:
            logits = tf.zeros((B, self.T))
            logs_u = tf.zeros((B, self.T))
            y_i = tf.zeros((B, ))
            for i in range(self.T):
                # (B, T)
                fill_mask = tf.concat([tf.zeros((B, i)),
                                       tf.ones((B, 1)),
                                       tf.zeros((B, self.T - (i + 1)))],
                                      axis=1)
                y = y + y_i[:, tf.newaxis] * fill_mask

                # (B, T, d_model), (B, H, T, F)
                dec_output_dict = self.decoder(x=y,
                                               enc_output=enc_output,
                                               training=training,
                                               look_ahead_mask=look_ahead_mask,
                                               padding_mask=dec_padding_mask)

                uncertainty_output = self.uncertainty_decoder(x=y,
                                                              training=training,
                                                              look_ahead_mask=look_ahead_mask)

                # (B, T, 1)
                logits_i = self.final_layer(dec_output_dict['x'])
                # (B, )
                logit_i = tf.squeeze(logits_i, axis=-1)[:, i]

                # (B, T, 1)
                log_u_i = self.uncertainty_dense(uncertainty_output)
                # (B, )
                log_u_i = tf.squeeze(log_u_i, axis=-1)[:, i]
                inv_u2_i = tf.math.exp(-2 * log_u_i)
                # (B, )
                prob_i = tf.math.sigmoid(logit_i * inv_u2_i)
                sample_i = tf.random.uniform(shape=(B, ),
                                             minval=0,
                                             maxval=1.0)

                # (B, )
                y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1
                # (B, T)
                logits = logits + logit_i[:, tf.newaxis] * fill_mask
                logs_u = logs_u + log_u_i[:, tf.newaxis] * fill_mask 


            # (B, T)
            invs_u2 = tf.math.exp(-2 * logs_u)
            preds = tf.math.sigmoid(logits * invs_u2)
            output_dict['log_u'] = logs_u
            output_dict['logits'] = logits
            output_dict['global_pred'] = preds
        return output_dict


class MultiTaskTransformer(tkm.Model):
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
                 use_labels=True,
                 **kwargs):
        super(MultiTaskTransformer, self).__init__()
        self.num_heads = num_heads
        self.encoding_compresser = tkl.Dense(units=d_model,
                                             activation='linear')
        self.d_model = d_model

        self.decoder = MultiTaskDecoder(num_layers=num_layers,
                                        d_model=d_model,
                                        num_heads=num_heads,
                                        dff=dff,
                                        T=T,
                                        temp_ty=temp_ty,
                                        temp_tx=temp_tx,
                                        rate=rate,
                                        use_encoder=use_encoder,
                                        use_labels=use_labels)
        self.num_layers = num_layers
        self.final_layer = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')
        self.T = T

    def call(self, inputs, y=None, training=None):
        output_dict = dict()
        B = tf.shape(inputs)[0]
        # (1, T, T) -> (B, T, T)
        tokens = tf.eye(self.T)[tf.newaxis, :, :] 
        # Add start to groundtruth sequence.
        if training:
            # (B, T + 1)
            y = tf.concat([tf.zeros((B, 1)), 2 * y - 1], axis=1)
        else:
            # (B, T + 1)
            y = tf.zeros((B, self.T + 1))

        # (B, N_p, N_p, d)
        x = self.encoding_compresser(inputs)
        # (B, q_x, d)
        x = tf.reshape(x, (B, -1, self.d_model))
 
        # (1, 1, T, T + 1) -> (B, H, T, T + 1)
        # Look ahead mask constrain tasks to look only at previous ground truths
        triangular_inferior = tf.linalg.band_part(tf.ones((self.T, self.T)), -1, 0) - tf.eye(self.T)
        look_ahead_mask = tf.concat([tf.ones((self.T, 1)),
                                     triangular_inferior],
                                     axis=-1)[tf.newaxis, tf.newaxis, :, :]

        if training:
            # (B, T, d_model)
            dec_output_dict = self.decoder(x=x,
                                           y=y,
                                           tokens=tokens,
                                           training=training,
                                           look_ahead_mask=look_ahead_mask)
            # (B, T, 1)
            dec_output = dec_output_dict['x']
            logits = self.final_layer(dec_output)
            # (B, T)
            logits = tf.squeeze(logits, axis=-1)
            output_dict['loss'] = logits
            output_dict['global_pred'] = tf.math.sigmoid(logits)
            for i in range(self.num_layers):
                output_dict['ty_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_ty'.format(i)]
                output_dict['tx_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_tx'.format(i)]

        else:
            logits = tf.zeros((B, self.T))
            y_i = tf.zeros((B, ))
            # (1, T) -> (B, T)
            fill_mask_logit = tf.zeros((self.T, ))[tf.newaxis, :]
            for i in range(self.T):
                # (1, T + 1) -> (B, T + 1)
                fill_mask_y = tf.concat([tf.zeros((1, 1)), fill_mask_logit], axis=1)
                y = y + y_i[:, tf.newaxis] * fill_mask_y

                # (B, T, d_model), (B, H, T, F)
                dec_output_dict = self.decoder(x=x,
                                               y=y,
                                               tokens=tokens,
                                               look_ahead_mask=look_ahead_mask)

                # (B, T, 1)
                logits_i = self.final_layer(dec_output_dict['x'])
                # (B, )
                logit_i = tf.squeeze(logits_i, axis=-1)[:, i]
                # (B, )
                prob_i = tf.math.sigmoid(logit_i)
                sample_i = tf.random.uniform(shape=(B, ),
                                             minval=0,
                                             maxval=1.0)

                # (B, )
                y_i = 2 * tf.dtypes.cast(prob_i - sample_i >= 0, tf.float32) - 1

                # (B, T)
                fill_mask_logit = tf.concat([tf.zeros((i, )),
                                             tf.ones((1, )),
                                             tf.zeros((self.T - (i + 1), ))],
                                            axis=0)[tf.newaxis, :]
                logits = logits + logit_i[:, tf.newaxis] * fill_mask_logit

            # (B, T)
            pred = tf.math.sigmoid(logits)
            output_dict['loss'] = logits
            output_dict['global_pred'] = pred
            for i in range(self.num_layers):
                output_dict['ty_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_ty'.format(i)]
                output_dict['tx_att_{}'.format(i)] = dec_output_dict['dec_layer{}_block_tx'.format(i)]
        return output_dict


SUPPORTED_ATTENTION_REGRESSORS = {"attention_regressor": AttentionRegressor,
                                  "uattention_regressor": UAttentionRegressor,
                                  "ruattention_regressor": RUAttentionRegressor,
                                  "mtt": MultiTaskTransformer}
