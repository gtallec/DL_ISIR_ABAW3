from models.layers.dense import ParallelDense

import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

class SoftLayerOrderingRegressor(tkm.Model):
    def __init__(self,
                 T, # Number of tasks
                 D, # Depth of the soft layer part
                 M, # Number of modules
                 M_units=64,
                 M_activations='relu',
                 M_use_bias=False,
                 **kwargs): # Size of the modules
        super(SoftLayerOrderingRegressor, self).__init__()
        self.T = T
        self.D = D
        self.M = M
        self.U = M_units

        # print('Modules')
        self.modules = ParallelDense(units=M_units,
                                     T=M,
                                     activation=M_activations,
                                     use_bias=M_use_bias)
        self.final_layer = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')
        self.sigma_logits = tf.Variable(tf.zeros((D, T, M)),
                                        trainable=True)

        self.input_compression = False
        self.input_compresser = tkl.Dense(units=M_units,
                                          activation='relu')

    def build(self, input_shape):
        self.input_compression = (input_shape[-1] != self.U)
        super(SoftLayerOrderingRegressor, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """
        inputs: (B, U)
        """
        # (B, T, U)
        outputs_dict = dict()
        input_shape = tf.shape(inputs)
        B = input_shape[0]

        if self.input_compression:
            inputs = self.input_compresser(inputs, training=training)

        T_inputs = tf.tile(tf.expand_dims(inputs, axis=1), (1, self.T, 1))

        for d in range(self.D):
            # (B, T, M, U)
            TM_inputs = tf.tile(tf.expand_dims(T_inputs, axis=-2), (1, 1, self.M, 1))
            # (B, T, M, U)
            TM_inputs = self.modules(TM_inputs, training=training)
            # (T, M)
            sigma_d = tf.nn.softmax(self.sigma_logits[d, :, :], axis=-1)
            # (B, T, M, U)
            sigma_d = tf.tile(tf.reshape(sigma_d, (1, self.T, self.M, 1)), (B, 1, 1, self.U))
            # (B, T, U)
            T_inputs = tf.math.reduce_sum(TM_inputs * sigma_d, axis=-2)

        # (B, T)
        logits = tf.squeeze(self.final_layer(T_inputs,
                                             training=training),
                            axis=-1)

        outputs_dict['sigma'] = tf.nn.softmax(self.sigma_logits, axis=-1)
        outputs_dict['global_pred'] = tf.math.sigmoid(logits)
        outputs_dict['loss'] = logits

        return outputs_dict
