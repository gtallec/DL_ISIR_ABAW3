import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm
import tensorflow as tf

from models.layers.dense import ParallelDense, ImbalancedParallelDense
import utils


class BTMRegressor(tkm.Model):
    def __init__(self, T, **kwargs):
        super(BTMRegressor, self).__init__() 
        self.mtb_matrix = utils.multilabel_to_binary_matrix(T)
        self.dense1 = tkl.Dense(units=512,
                                activation='relu')
        self.bn1 = tkl.BatchNormalization()
        self.dense2 = tkl.Dense(units=512,
                                activation='relu')
        self.bn2 = tkl.BatchNormalization()
        self.dense3 = tkl.Dense(units=2 ** T,
                                activation='linear')
        self.T = T

    def call(self, x, training=None, **kwargs):
        outputs_dict = dict()
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dense3(x)

        outputs_dict['logits'] = x
        exp_x = tf.math.exp(x)
        x = exp_x / (tf.tile(tf.expand_dims(tf.math.reduce_sum(exp_x, axis=1), axis=1),
                             multiples=[1, 2 ** (self.T)]))
        outputs_dict['prediction'] = tf.matmul(x, tf.transpose(self.mtb_matrix))
        return outputs_dict

class VMNS(tkm.Model):
    def __init__(self, T, units, imbalance=False, frequencies=None, **kwargs):
        self.T = T
        super(VMNS, self).__init__()

        self.pdense1 = ParallelDense(units=units,
                                     T=T,
                                     activation='relu')
        self.bn1 = tkl.BatchNormalization(axis=(1, 2))

        self.pdense2 = ParallelDense(units=units,
                                     T=T,
                                     activation='relu')
        self.bn2 = tkl.BatchNormalization(axis=(1, 2))

        if imbalance:
            tf.print("frequencies : ", frequencies)
            self.pdense3 = ImbalancedParallelDense(T=T,
                                                   frequencies=frequencies,
                                                   activation='linear')
        else:
            self.pdense3 = ParallelDense(units=1,
                                         T=T,
                                         activation='linear')


    def call(self, x, training=None, **kwargs):
        # (B, T, I)
        x_tiled = tf.tile(tf.expand_dims(x, axis=-2), multiples=[1, self.T, 1])

        # (B, T, U)
        reg = self.pdense1(x_tiled)
        # tf.print("reg1 has nan : ", tf.math.reduce_any(tf.math.is_nan(reg)))

        reg = self.bn1(reg, training=training)
        # tf.print("bn1 has nan : ", tf.math.reduce_any(tf.math.is_nan(reg)))

        # (B, T, U)
        reg = self.pdense2(reg)
        # tf.print("reg2 has nan : ", tf.math.reduce_any(tf.math.is_nan(reg)))

        reg = self.bn2(reg, training=training)
        # tf.print("bn2 has nan : ", tf.math.reduce_any(tf.math.is_nan(reg)))

        # (B, T, 1)
        reg = self.pdense3(reg)
        # tf.print("reg3 has nan : ", tf.math.reduce_any(tf.math.is_nan(reg)))

        # (B, T)
        reg = tf.squeeze(reg, axis=-1)
        output_dict = dict()
        output_dict['global_pred'] = tf.math.sigmoid(reg)
        output_dict['loss'] = reg
        # tf.print("reg.shape : ", reg.shape)
        # tf.print("result has nan : ", tf.math.reduce_any(tf.math.is_nan(reg)))
        return output_dict

class VMNC(tkm.Model):
    def __init__(self, units, T, **kwargs):
        super(VMNC, self).__init__()

        self.dense1 = tkl.Dense(units=units, activation='relu')
        self.dense2 = tkl.Dense(units=units, activation='relu')
        self.dense3 = tkl.Dense(units=T, activation='linear')

        self.bn1 = tkl.BatchNormalization()
        self.bn2 = tkl.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        output_dict = dict()
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)

        x = self.dense2(x)
        x = self.bn2(x, training=training)

        output = self.dense3(x, training=training)
        output_dict['global_pred'] = tf.math.sigmoid(output)
        output_dict['loss'] = output
        return output_dict

class DenseRegressor(tkm.Model):
    def __init__(self, units, T, **kwargs):
        super(DenseRegressor, self).__init__()
        self.dense1 = tkl.Dense(units=units,
                                activation='relu')
        self.bn1 = tkl.BatchNormalization()
        self.dense2 = tkl.Dense(units=units,
                                activation='relu')

        self.bn2 = tkl.BatchNormalization()
        self.dense3 = tkl.Dense(units=T,
                                activation='linear')

    def call(self, x, training=None, **kwargs):
        reg = self.dense1(x)
        reg = self.bn1(reg, training=training)
        reg = self.dense2(reg)
        reg = self.bn2(reg, training=training)
        reg = self.dense3(reg)
        return reg

class TVMNS(tkm.Model):
    def __init__(self, T, S, units, **kwargs):
        super(TVMNS, self).__init__()

        self.dense1 = ParallelDense(units=units,
                                    T=T,
                                    activation='relu')
        self.bn1 = tkl.BatchNormalization(axis=(1, 2))

        self.dense2 = ParallelDense(units=units,
                                    T=T,
                                    activation='relu')
        self.bn2 = tkl.BatchNormalization(axis=(1, 2))

        self.dense3 = ParallelDense(units=S,
                                    T=T,
                                    activation='linear')
        self.T = T

    def call(self, x, padding_mask=None, training=None, **kwargs):
        """ x of size (B, H) """
        output_dict = dict()
        # (B, T, H)
        x_tiled = tf.tile(x[:, tf.newaxis, :], multiples=(1, self.T, 1))
        # (B, T, U)
        x_tiled = self.dense1(x_tiled)
        # (B, T, U)
        # x_tiled = self.bn1(x_tiled, training=False)
        # (B, T, U)
        x_tiled = self.dense2(x_tiled)
        # (B, T, U)
        # x_tiled = self.bn2(x_tiled, training=False)
        # (B, T, S)
        logits = self.dense3(x_tiled)
        # (B, S, T)
        logits = tf.transpose(logits, perm=(0, 2, 1))
        output_dict['global_pred'] = dict()
        output_dict['global_pred']['prediction'] = tf.math.sigmoid(logits)
        output_dict['global_pred']['padding_mask'] = padding_mask

        output_dict['loss'] = dict()
        output_dict['loss']['logits'] = logits
        output_dict['loss']['padding_mask'] = padding_mask
        return output_dict


SUPPORTED_DENSE_REGRESSORS = {"btm_regressor": BTMRegressor,
                              "tvmns": TVMNS,
                              "vmns": VMNS,
                              "vmnc": VMNC}
