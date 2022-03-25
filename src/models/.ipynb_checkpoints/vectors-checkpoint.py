import copy as cp

import tensorflow as tf
import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl

from models.layers.recurrent_cells import recurrent_cell

class RecurrentMixture(tkm.Model):
    " A recurrent_cell for all timesteps "
    def __init__(self, T, recurrent_cell_args, **kwargs):
        super(RecurrentMixture, self).__init__(**kwargs)
        self.recurrent_cell = recurrent_cell(recurrent_cell_args)
        self.dense = tkl.Dense(units=T,
                               activation='linear')
        self.units = recurrent_cell_args['units']

    def call(self, inputs, states, training=None, **kwargs):
        (output, states) = self.recurrent_cell(inputs=inputs,
                                               states=states,
                                               training=training)

        logit = self.dense(output, training=training)
        return logit, states[0]
    
    def get_units(self):
        return self.units
        
class RecurrentMixturev2(tkm.Model):
    "A recurrent_cell by timestep "
    def __init__(self, T, recurrent_cell_args, **kwargs):
        super(RecurrentMixturev2, self).__init__(**kwargs)
        self.recurrent_cells = []
        self.denses = []

        for i in range(T):
            self.recurrent_cells.append(recurrent_cell(cp.deepcopy(recurrent_cell_args)))
            self.denses.append(tkl.Dense(units=T,
                                         activation='linear'))

        self.units = recurrent_cell_args['units']

    def call(self, inputs, states, timestep, training=None, **kwargs):
        (output, states) = self.recurrent_cells[timestep](inputs=inputs,
                                                          states=states,
                                                          training=training)

        logit = self.denses[timestep](output, training=training)
        return logit, states[0]
    
    def get_units(self):
        return self.units

class XMixture(tkm.Model):
    def __init__(self, P, hidden_layers, T=1.0, batchnorm=False, units=128, uniform_init=False, **kwargs):
        super(XMixture, self).__init__(**kwargs)

        self.bns = []
        self.denses = []
        self.batchnorm = batchnorm
        self.temperature = T

        for i in range(hidden_layers):
            self.denses.append(tkl.Dense(units=units, activation='relu'))

            if batchnorm:
                self.bns.append(tkl.BatchNormalization())

        if uniform_init:
            last_kernel_initializer = "zeros"
        else:
            last_kernel_initializer = "glorot_uniform"

        self.final_dense = tkl.Dense(units=P,
                                     kernel_initializer=last_kernel_initializer,
                                     activation='linear')

    def call(self, x, training=None, **kwargs):
        output_dict = dict()
        output_dict['track_grad'] = dict()
        x_enc = x
        for i in range(len(self.denses)):
            x_enc = self.denses[i](x_enc)
            if self.batchnorm:
                x_enc = self.bn1(x_enc, training=training)
            output_dict['track_grad']['mixture_layer_{}'.format(i)] = x_enc

        x_enc = self.final_dense(x_enc, training=training)
        output_dict['track_grad']['mixture_logits'] = x_enc
        output_dict['logits'] = x_enc / self.temperature
        output_dict['mixture_kernel'] = self.final_dense.kernel
        output_dict['mixture_bias'] = self.final_dense.bias
        return output_dict

class NetL1BallVector(tkm.Model):
    def __init__(self, n_permutations, temperature=1.0, **kwargs):
        super(NetL1BallVector, self).__init__(**kwargs)

        self.dense1 = tkl.Dense(units=256,
                                activation='relu')

        self.dense2 = tkl.Dense(units=256,
                                activation='relu')

        self.dense3 = tkl.Dense(units=n_permutations,
                                activation='linear')
        self.T = temperature

    def call(self, x, **kwargs):
        x_input = tf.ones((1, 256))
        x_net = self.dense1(x_input)
        x_net = self.dense2(x_net)
        x_net = self.dense3(x_net)
        exp_x = tf.math.exp(self.vector/self.temperature)
        output = exp_x/tf.math.reduce_sum(exp_x)
        return tf.squeeze(output, axis=0)

class L1BallVector(tkm.Model):
    def __init__(self, n_permutations, temperature=1.0, name='l1ballvector/mixture', **kwargs):
        super(L1BallVector, self).__init__(**kwargs)
        self.vector = tf.Variable(tf.zeros(n_permutations,),
                                  dtype=tf.float32,
                                  name=name,
                                  trainable=True)
        self.temperature = temperature

    def call(self, inputs):
        exp_x = tf.math.exp(self.vector/self.temperature)
        return exp_x/tf.math.reduce_sum(exp_x)

class Vector(tkm.Model):
    def __init__(self, n_permutations, name='vector/mixture', **kwargs):
        super(Vector, self).__init__(**kwargs)
        self.vector = tf.Variable(tf.ones(n_permutations,),
                                  dtype=tf.float32,
                                  name=name,
                                  trainable=True)

    def call(self, inputs):
        return self.vector

class VectorList(tkm.Model):
    def __init__(self, n_permutations_list, name='vectorlist', **kwargs):
        super(VectorList, self).__init__(**kwargs)
        self.vectors = []
        for i in range(len(n_permutations_list)):
            self.vectors.append(Vector(n_permutations=n_permutations_list[i], name='vectorlist/mixture_{}'.format(i)))

    def call(self, inputs, **kwargs):
        pass

    def get_vector(self, i):
        return self.vectors[i]

class L1BallVectorList(tkm.Model):
    def __init__(self, n_perms, temperature=1.0, **kwargs):
        super(L1BallVectorList, self).__init__(**kwargs)
        self.vectors = []
        for i in range(len(n_perms)):
            self.vectors.append(L1BallVector(n_permutations=n_perms[i], temperature=temperature, name="l1ballvectorlist/mixture_{}".format(i)))
        self.build((None, None))

    def call(self, inputs):
        vectors = []
        for vector in self.vectors:
            vectors.append(vector(None))
        return vectors

    def get_vectors(self):
        return self.vectors


SUPPORTED_VECTORS = {"l1ballvector": L1BallVector,
                     "l1ballvectorlist": L1BallVectorList,
                     "xmixture": XMixture,
                     "vector": Vector,
                     "vectorlist": VectorList,
                     "rec_mixture": RecurrentMixture,
                     "rec_mixturev2": RecurrentMixturev2}
