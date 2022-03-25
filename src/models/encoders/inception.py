import tensorflow.keras.models as tkm
import tensorflow.keras.layers as tkl
import os

import tensorflow as tf

from models.layers.normalisation_layers import InceptionNormalisation
####################################################################################################
#                                          Block 35                                                #
####################################################################################################
class Block35_Branch_0(tkm.Model):

    def __init__(self, batch_norm={}, regularization={}, name='Branch_0', **kwargs):
        super(Block35_Branch_0, self).__init__(name=name, **kwargs)
        self.conv = tkl.Conv2D(filters=32,
                               kernel_size=1,
                               padding='same',
                               activation='relu',
                               use_bias=False,
                               **regularization,
                               name='Conv2d_1x1')
        self.bn = tkl.BatchNormalization(**batch_norm,
                                         name='Conv2d_1x1' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv(x)
        tower_conv = self.bn(tower_conv, training=training)
        return tower_conv


class Block35_Branch_1(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, name='Branch_1', **kwargs):
        super(Block35_Branch_1, self).__init__(name=name, **kwargs)
        self.conv1 = tkl.Conv2D(filters=32,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv2 = tkl.Conv2D(filters=32,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0b_3x3')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0b_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv1(x)
        tower_conv = self.bn1(tower_conv, training=training)
        tower_conv = self.conv2(tower_conv)
        tower_conv = self.bn2(tower_conv, training=training)
        return tower_conv


class Block35_Branch_2(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, name='Branch_2', **kwargs):
        super(Block35_Branch_2, self).__init__(name=name, **kwargs)
        self.conv1 = tkl.Conv2D(filters=32,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv2 = tkl.Conv2D(filters=32,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0b_3x3')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0b_3x3' + '/' + 'BatchNorm')
        self.conv3 = tkl.Conv2D(filters=32,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0c_3x3')
        self.bn3 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0c_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv1(x)
        tower_conv = self.bn1(tower_conv, training=training)
        tower_conv = self.conv2(tower_conv)
        tower_conv = self.bn2(tower_conv, training=training)
        tower_conv = self.conv3(tower_conv)
        tower_conv = self.bn3(tower_conv, training=training)
        return tower_conv


class Block35(tkm.Model):
    def __init__(self, scale=1.0, activation='relu', batch_norm={}, regularization={}, **kwargs):
        super(Block35, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Block35_Branch_0(batch_norm=batch_norm,
                                         regularization=regularization)
        self.branch_1 = Block35_Branch_1(batch_norm=batch_norm,
                                         regularization=regularization)
        self.branch_2 = Block35_Branch_2(batch_norm=batch_norm,
                                         regularization=regularization)
        self.mixed = tkl.Concatenate(axis=3)
        self.up = tkl.Conv2D(filters=256,
                             kernel_size=1,
                             name='Conv2d_1x1',
                             **regularization)
        self.net_output = tkl.Add()

    def call(self, inputs, training=None):
        tower_conv0 = self.branch_0(inputs, training=training)
        tower_conv1 = self.branch_1(inputs, training=training)
        tower_conv2 = self.branch_2(inputs, training=training)
        mixed = self.mixed([tower_conv0, tower_conv1, tower_conv2])
        up = self.up(mixed)
        output = self.net_output([inputs, self.scale * up])
        if self.activation is not None:
            output = tkl.Activation(self.activation)(output)
        return output

####################################################################################################
#                                          Block 17                                                #
####################################################################################################

class Block17_Branch_0(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, name='Branch_0', **kwargs):
        super(Block17_Branch_0, self).__init__(name=name, **kwargs)
        self.conv = tkl.Conv2D(filters=128,
                               kernel_size=1,
                               padding='same',
                               activation='relu',
                               use_bias=False,
                               **regularization,
                               name='Conv2d_1x1')
        self.bn = tkl.BatchNormalization(**batch_norm,
                                         name='Conv2d_1x1' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv(x)
        tower_conv = self.bn(tower_conv, training=training)
        return tower_conv


class Block17_Branch_1(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, name='Branch_1', **kwargs):
        super(Block17_Branch_1, self).__init__(name=name, **kwargs)
        self.conv0 = tkl.Conv2D(filters=128,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv1 = tkl.Conv2D(filters=128,
                                kernel_size=(1, 7),
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0b_1x7')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0b_1x7' + '/' + 'BatchNorm')
        self.conv2 = tkl.Conv2D(filters=128,
                                kernel_size=(7, 1),
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0c_7x1')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0c_7x1' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x)
        tower_conv = self.bn0(tower_conv, training=training)
        tower_conv = self.conv1(tower_conv)
        tower_conv = self.bn1(tower_conv, training=training)
        tower_conv = self.conv2(tower_conv)
        tower_conv = self.bn2(tower_conv, training=training)
        return tower_conv


class Block17(tkm.Model):
    def __init__(self, scale=1.0, activation='relu', batch_norm={}, regularization={}, **kwargs):
        super(Block17, self).__init__()
        self.activation = activation
        self.scale = scale
        self.branch_0 = Block17_Branch_0(batch_norm=batch_norm,
                                         regularization=regularization)
        self.branch_1 = Block17_Branch_1(batch_norm=batch_norm,
                                         regularization=regularization)
        self.mixed = tkl.Concatenate(axis=3)
        self.up = tkl.Conv2D(filters=896,
                             kernel_size=1,
                             **regularization,
                             name='Conv2d_1x1')
        self.net_output = tkl.Add()

    def call(self, inputs, training=None):
        tower_conv0 = self.branch_0(inputs, training=training)
        tower_conv1 = self.branch_1(inputs, training=training)
        mixed = self.mixed([tower_conv0, tower_conv1])
        up = self.up(mixed)
        output = self.net_output([inputs, self.scale * up])
        if self.activation is not None:
            output = tkl.Activation(self.activation)(output)
        return output

####################################################################################################
#                                           Block 8                                                #
####################################################################################################

class Block8_Branch_0(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, name='Branch_0', **kwargs):
        super(Block8_Branch_0, self).__init__(name=name, **kwargs)
        self.conv = tkl.Conv2D(filters=192,
                               kernel_size=1,
                               padding='same',
                               activation='relu',
                               use_bias=False,
                               **regularization,
                               name='Conv2d_1x1')
        self.bn = tkl.BatchNormalization(**batch_norm,
                                         name='Conv2d_1x1' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv(x)
        tower_conv = self.bn(tower_conv, training=training)
        return tower_conv


class Block8_Branch_1(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, name='Branch_1', **kwargs):
        super(Block8_Branch_1, self).__init__(name=name, **kwargs)
        self.conv0 = tkl.Conv2D(filters=192,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv1 = tkl.Conv2D(filters=192,
                                kernel_size=(1, 3),
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0b_1x3')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0b_1x3' + '/' + 'BatchNorm')
        self.conv2 = tkl.Conv2D(filters=192,
                                kernel_size=(3, 1),
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0c_3x1')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0c_3x1' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x)
        tower_conv = self.bn0(tower_conv, training=training)
        tower_conv = self.conv1(tower_conv)
        tower_conv = self.bn1(tower_conv, training=training)
        tower_conv = self.conv2(tower_conv)
        tower_conv = self.bn2(tower_conv, training=training)
        return tower_conv


class Block8(tkm.Model):
    def __init__(self, scale=1.0, activation='relu', batch_norm={}, regularization={}, **kwargs):
        super(Block8, self).__init__()
        self.activation = activation
        self.scale = scale
        self.branch_0 = Block8_Branch_0(batch_norm=batch_norm,
                                        regularization=regularization)
        self.branch_1 = Block8_Branch_1(batch_norm=batch_norm,
                                        regularization=regularization)
        self.mixed = tkl.Concatenate(axis=3)

        self.up = tkl.Conv2D(filters=1792,
                             kernel_size=1,
                             name='Conv2d_1x1')
        self.net_output = tkl.Add()

    def call(self, inputs, training=None):
        tower_conv0 = self.branch_0(inputs, training=training)
        tower_conv1 = self.branch_1(inputs, training=training)
        mixed = self.mixed([tower_conv0, tower_conv1])
        up = self.up(mixed)
        output = self.net_output([inputs, self.scale * up])
        if self.activation is not None:
            output = tkl.Activation(self.activation)(output)
        return output

####################################################################################################
#                                       Reduction_A                                                #
####################################################################################################

class Reduction_A_Branch_0(tkm.Model):
    def __init__(self, n, batch_norm={}, name='Branch_0', regularization={}, **kwargs):
        super(Reduction_A_Branch_0, self).__init__(**kwargs, name=name)
        self.conv0 = tkl.Conv2D(filters=n,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_1a_3x3')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_1a_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x, training=training)
        tower_conv = self.bn0(tower_conv, training=training)
        return tower_conv


class Reduction_A_Branch_1(tkm.Model):
    def __init__(self, k, l, m, name='Branch_1', batch_norm={}, regularization={}, **kwargs):
        super(Reduction_A_Branch_1, self).__init__(**kwargs, name=name)
        self.conv0 = tkl.Conv2D(filters=k,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv1 = tkl.Conv2D(filters=l,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0b_3x3')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0b_3x3' + '/' + 'BatchNorm')
        self.conv2 = tkl.Conv2D(filters=m,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_1a_3x3')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_1a_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x)
        tower_conv = self.bn0(tower_conv, training=training)
        tower_conv = self.conv1(tower_conv)
        tower_conv = self.bn1(tower_conv, training=training)
        tower_conv = self.conv2(tower_conv)
        tower_conv = self.bn2(tower_conv, training=training)
        return tower_conv


class Reduction_A_Branch_2(tkm.Model):

    def __init__(self, name='Branch_2', **kwargs):
        super(Reduction_A_Branch_2, self).__init__(**kwargs, name=name)
        self.pool0 = tkl.MaxPooling2D(pool_size=2,
                                      strides=2,
                                      padding='valid',
                                      name='MaxPool_2a_3x3')

    def call(self, x, training=None):
        pool = self.pool0(x, training=training)
        return pool


class Reduction_A(tkm.Model):
    def __init__(self, k, l, m, n, batch_norm={}, regularization={}, **kwargs):
        super(Reduction_A, self).__init__()
        self.branch_0 = Reduction_A_Branch_0(n=n,
                                             batch_norm=batch_norm,
                                             regularization=regularization)
        self.branch_1 = Reduction_A_Branch_1(k=k, l=l, m=m,
                                             batch_norm=batch_norm,
                                             regularization=regularization)
        self.branch_2 = Reduction_A_Branch_2()
        self.net_output = tkl.Concatenate(axis=3)

    def call(self, inputs, training=None):
        tower_conv0 = self.branch_0(inputs, training=training)
        tower_conv1 = self.branch_1(inputs, training=training)
        tower_pool = self.branch_2(inputs, training=training)
        output = self.net_output([tower_conv0, tower_conv1, tower_pool])
        return output

####################################################################################################
#                                       Reduction_B                                                #
####################################################################################################

class Reduction_B_Branch_0(tkm.Model):
    def __init__(self, batch_norm={}, name='Branch_0', regularization={}, **kwargs):
        super(Reduction_B_Branch_0, self).__init__(**kwargs, name=name)
        self.conv0 = tkl.Conv2D(filters=256,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv1 = tkl.Conv2D(filters=384,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_1a_3x3')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_1a_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x)
        tower_conv = self.bn0(tower_conv, training=training)
        tower_conv = self.conv1(tower_conv)
        tower_conv = self.bn1(tower_conv, training=training)
        return tower_conv


class Reduction_B_Branch_1(tkm.Model):
    def __init__(self, batch_norm={}, name='Branch_1', regularization={}, **kwargs):
        super(Reduction_B_Branch_1, self).__init__(**kwargs, name=name)
        self.conv0 = tkl.Conv2D(filters=256,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv1 = tkl.Conv2D(filters=256,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_1a_3x3')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_1a_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x)
        tower_conv = self.bn0(tower_conv, training=training)
        tower_conv = self.conv1(tower_conv)
        tower_conv = self.bn1(tower_conv, training=training)
        return tower_conv


class Reduction_B_Branch_2(tkm.Model):
    def __init__(self, batch_norm={}, name='Branch_2', regularization={}, **kwargs):
        super(Reduction_B_Branch_2, self).__init__(**kwargs, name=name)
        self.conv0 = tkl.Conv2D(filters=256,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0a_1x1')
        self.bn0 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0a_1x1' + '/' + 'BatchNorm')
        self.conv1 = tkl.Conv2D(filters=256,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_0b_3x3')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_0b_3x3' + '/' + 'BatchNorm')
        self.conv2 = tkl.Conv2D(filters=256,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_1a_3x3')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_1a_3x3' + '/' + 'BatchNorm')

    def call(self, x, training=None):
        tower_conv = self.conv0(x)
        tower_conv = self.bn0(tower_conv, training=training)
        tower_conv = self.conv1(tower_conv)
        tower_conv = self.bn1(tower_conv, training=training)
        tower_conv = self.conv2(tower_conv)
        tower_conv = self.bn2(tower_conv, training=training)
        return tower_conv


class Reduction_B_Branch_3(tkm.Model):
    def __init__(self, batch_norm={}, name='Branch_3', regularization={}, **kwargs):
        super(Reduction_B_Branch_3, self).__init__(**kwargs, name=name)
        self.pool0 = tkl.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='valid',
                                      name='MaxPool_3a_3x3')

    def call(self, x, training=None):
        tower_pool = self.pool0(x)
        return tower_pool


class Reduction_B(tkm.Model):
    def __init__(self, batch_norm={}, regularization={}, **kwargs):
        super(Reduction_B, self).__init__()
        self.branch_0 = Reduction_B_Branch_0(batch_norm=batch_norm,
                                             regularization=regularization)
        self.branch_1 = Reduction_B_Branch_1(batch_norm=batch_norm,
                                             regularization=regularization)
        self.branch_2 = Reduction_B_Branch_2(batch_norm=batch_norm,
                                             regularization=regularization)
        self.branch_3 = Reduction_B_Branch_3(batch_norm=batch_norm,
                                             regularization=regularization)
        self.net_output = tkl.Concatenate(axis=3)

    def call(self, inputs, training=None):
        tower_conv0 = self.branch_0(inputs, training=training)
        tower_conv1 = self.branch_1(inputs, training=training)
        tower_conv2 = self.branch_2(inputs, training=training)
        tower_pool = self.branch_3(inputs)
        output = self.net_output([tower_conv0, tower_conv1, tower_conv2, tower_pool])
        return output

####################################################################################################
#                                        Repeat                                                    #
####################################################################################################

class Repeat(tkm.Model):
    def __init__(self, model, n_repeat, name_template, name='Repeat', **kwargs):
        super(Repeat, self).__init__(name=name)
        self.models = [model(**kwargs, name=name_template.format(i + 1)) for i in range(n_repeat)]

    def call(self, inputs, training=None):
        repeat_input = inputs
        for model in self.models:
            repeat_input = model(repeat_input, training=training)
        return repeat_input

####################################################################################################
#                                        Inception                                                 #
####################################################################################################

class InceptionResnet(tkm.Model):
    def __init__(self, bottleneck_layer_size,
                 regularization={}, name='InceptionResnetV1',
                 batch_norm={}, attention=False, **kwargs):
        super(InceptionResnet, self).__init__(name=name)
        self.normalisation_layer = InceptionNormalisation()
        self.attention = attention
        self.bottleneck_layer_size = bottleneck_layer_size
        # ------------------------------------------------------------------------- #
        #                                Stem                                       #
        # ------------------------------------------------------------------------- #
        # 149 x 149 x 32
        self.conv1 = tkl.Conv2D(filters=32,
                                kernel_size=3,
                                strides=2,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_1a_3x3')
        self.bn1 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_1a_3x3' + '/' + 'BatchNorm')
        # 147 x 147 x 32
        self.conv2 = tkl.Conv2D(filters=32,
                                kernel_size=3,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_2a_3x3')
        self.bn2 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_2a_3x3' + '/' + 'BatchNorm')
        # 147 x 147 x 64
        self.conv3 = tkl.Conv2D(filters=64,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_2b_3x3')
        self.bn3 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_2b_3x3' + '/' + 'BatchNorm')
        # 73 x 73 x 64
        self.pool1 = tkl.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='valid',
                                      name='MaxPool_3a_3x3')
        # 73 x 73 x 80
        self.conv4 = tkl.Conv2D(filters=80,
                                kernel_size=1,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_3b_1x1')
        self.bn4 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_3b_1x1' + '/' + 'BatchNorm')
        # 71 x 71 x 192
        self.conv5 = tkl.Conv2D(filters=192,
                                kernel_size=3,
                                padding='valid',
                                activation='relu',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_4a_3x3')
        self.bn5 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_4a_3x3' + '/' + 'BatchNorm')
        # 35 x 35 x 256
        self.conv6 = tkl.Conv2D(filters=256,
                                kernel_size=3,
                                strides=2,
                                activation='relu',
                                padding='valid',
                                use_bias=False,
                                **regularization,
                                name='Conv2d_4b_3x3')
        self.bn6 = tkl.BatchNormalization(**batch_norm,
                                          name='Conv2d_4b_3x3' + '/' + 'BatchNorm')
        # ------------------------------------------------------------------------- #
        #                                  Core                                     #
        # ------------------------------------------------------------------------- #
        self.repeat = Repeat(model=Block35,
                             n_repeat=5,
                             scale=0.17,
                             batch_norm=batch_norm,
                             regularization=regularization,
                             name_template='block35_{}',
                             name='Repeat')
        self.reduction_a = Reduction_A(192, 192, 256, 384,
                                       batch_norm=batch_norm,
                                       regularization=regularization,
                                       name='Mixed_6a')
        self.repeat_1 = Repeat(model=Block17,
                               n_repeat=10,
                               scale=0.10,
                               batch_norm=batch_norm,
                               regularization=regularization,
                               name_template='block17_{}',
                               name='Repeat_1')
        self.reduction_b = Reduction_B(batch_norm=batch_norm,
                                       regularization=regularization,
                                       name='Mixed_7a')
        self.repeat_2 = Repeat(model=Block8,
                               n_repeat=5,
                               scale=0.20,
                               batch_norm=batch_norm,
                               regularization=regularization,
                               name_template='block8_{}',
                               name='Repeat_2')
        self.inception_c = Block8(activation=None,
                                  batch_norm=batch_norm,
                                  regularization=regularization,
                                  name='Block8')
        self.pool2 = tkl.AveragePooling2D(pool_size=(8, 8),
                                          padding='same',
                                          name='AvgPool_1a_8x8')
        # ------------------------------------------------------------------------- #
        #                             Regression                                    #
        # ------------------------------------------------------------------------- #
        self.flatten1 = tkl.Flatten()
        self.dense1 = tkl.Dense(units=bottleneck_layer_size,
                                activation='relu',
                                **regularization,
                                use_bias=False,
                                name='Bottleneck')
        self.bn7 = tkl.BatchNormalization(**batch_norm,
                                          name='Bottleneck/BatchNorm')

    def build(self, input_shape):
        if self.attention:
            self.dense1.build((None, 1792))
            self.bn7.build((None, self.bottleneck_layer_size))
        super(InceptionResnet, self).build(input_shape)


    def call(self, inputs, training=None, attention=None, **kwargs):

        # ------------------------------------------------------------------------- #
        #                                Stem                                       #
        # ------------------------------------------------------------------------- #
        # 149 x 149 x 32
        # print('inputs: ', inputs.shape)
        inputs = self.normalisation_layer(inputs)
        x = self.conv1(inputs)
        # print('conv (1) : ', x.shape)
        x = self.bn1(x, training=training)
        # 147 x 147 x 32
        x = self.conv2(x)
        # print('conv (2) : ', x.shape)
        x = self.bn2(x, training=training)
        # 147 x 147 x 64
        x = self.conv3(x)
        # print('conv (3) : ', x.shape)
        x = self.bn3(x, training=training)
        # 73 x 73 x 64
        x = self.pool1(x)
        # print('conv (4) : ', x.shape)
        # 73 x 73 x 80
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        # print('conv (5) : ', x.shape)
        # 71 x 71 x 192
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        # print('conv (6) : ', x.shape)
        # 35 x 35 x 256
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        # print('conv (7) : ', x.shape)
        # ------------------------------------------------------------------------- #
        #                                  Core                                     #
        # ------------------------------------------------------------------------- #
        x = self.repeat(x, training=training)
        x = self.reduction_a(x, training=training)
        x = self.repeat_1(x, training=training)
        x = self.reduction_b(x, training=training)
        x = self.repeat_2(x, training=training)
        x = self.inception_c(x, training=training)
        if not(self.attention):
            # print('before pooling : ', x.shape)
            x = self.pool2(x)
            # print('before flattening : ', x.shape)
            x = self.flatten1(x)
            # ------------------------------------------------------------------------- #
            #                             Regression                                    #
            # ------------------------------------------------------------------------- #
            x = self.dense1(x)
            x = self.bn7(x, training=training)
        else:
            batchsize = tf.shape(x)[0]
            x = tf.reshape(x, (batchsize, 9, 1792))
        return x

class Inceptionv3(tkm.Model):
    def __init__(self, pooling=None, weights=None, **kwargs):
        super(Inceptionv3, self).__init__()
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                           weights=weights,
                                                                           pooling=pooling)

    def call(self, inputs, training=None, **kwargs):
        x = tf.keras.applications.inception_v3.preprocess_input(255 * inputs)
        return self.inception_v3(x, training=training)

class Inceptionv3Attention(tkm.Model):
    def __init__(self, d_model, weights=None, **kwargs):
        super(Inceptionv3Attention, self).__init__()
        tf.print("weights : ", weights)
        self.d_model = d_model
        self.inception_v3 = Inceptionv3(pooling=None,
                                        weights=weights)
        self.compresser = tkl.Dense(units=d_model)

    def call(self, inputs, training=None, **kwargs):
        B = tf.shape(inputs)[0]
        # (B, N_patch, N_patch, dim_f)
        x = self.inception_v3(inputs, training=training)
        x = tf.reshape(x, (B, 64, 2048))
        # (B, 64, d_model)
        x = self.compresser(x)
        return x






if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    weights = os.path.join('..', '..', '..', 'resources', 'pretrained_weights', 'weights.h5')
    model = InceptionResnet(512, attention=False)
    model.build((None, 160, 160, 3))
    model.load_weights(weights)
    # model.summary()
