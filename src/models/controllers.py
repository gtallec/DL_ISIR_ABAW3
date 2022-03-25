import copy as cp
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl



class Controller(tkm.Model):
    def __init__(self, d_model, **kwargs):
        super(Controller, self).__init__()
        self.dense_controller = tkl.Dense(units=d_model,
                                          activation='linear',
                                          kernel_initializer='zeros',
                                          bias_initializer='zeros')

    def call(self, inputs, training=None, **kwargs):
        return self.dense(inputs)

class Controllers(tkm.Model):
    def __init__(self, d_model, num_layers, control_name='x', controller_init='zeros', **kwargs):
        super(Controllers, self).__init__()
        self.dense_controllers = [tkl.Dense(units=d_model,
                                            activation='linear',
                                            kernel_initializer=controller_init,
                                            bias_initializer='zeros',
                                            name='controllers_{}/controller_{}'.format(control_name, i))
                                  for i in range(num_layers)]

    def call(self, inputs, training=None, **kwargs):
        pass

    def get(self, i):
        return self.dense_controllers[i]
