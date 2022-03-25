import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

class ToyRegressor(tkm.Model):
    def __init__(self, n_task, batch_norm, **kwargs):
        super(ToyRegressor, self).__init__(**kwargs)

        self.n_task = n_task
        self.dense1 = tkl.Dense(units=256,
                                activation='relu')
        self.bn1 = tkl.BatchNormalization(**batch_norm)
        self.dense2 = tkl.Dense(units=n_task,
                                activation='sigmoid')

    def call(self, x, training=None, **kwargs):
        x_enc = self.dense1(x)
        x_enc = self.bn1(x_enc, training=training)
        x_enc = self.dense2(x_enc)
        output = dict()
        output['output'] = x_enc
        return output


SUPPORTED_TOY_REGRESSORS = {"toy_regressor": ToyRegressor}
