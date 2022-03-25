import tensorflow.keras.layers as tkl
import tensorflow as tf

from models.layers.sa_layers import SALayers
from models.layers.ca_layers import MCALayers
from models.layers.dense import ParallelDense


class MultiLabelEncoder(tkl.Layer):
    def __init__(self,
                 d_model,
                 T):
        super(MultiLabelEncoder, self).__init__()
        kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.dense = ParallelDense(units=d_model,
                                   T=T,
                                   activation='linear',
                                   kernel_initializer=kernel_initializer)

    def call(self, x, training=None):
        """
        x (B, T): The binary labels to encode.
        -1 : not activated,
        0 : unknown,
        1 : activated
        """
        # (B, T, 3)
        x_one_hot = tf.gather(tf.eye(3), tf.dtypes.cast(x + 1, tf.int32), axis=0)
        return self.dense(x_one_hot)


class TMCALayers(tkl.Layer):
    """ Token based Multi Cross Attention Layers with Image and Labels"""
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 temp_tx=1.0,
                 temp_ty=1.0,
                 rate=0.1,
                 ca_order="xy"):

        super(TMCALayers, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.ca_order = ca_order

        if self.ca_order == "xy":
            temps = [temp_tx, temp_ty]
        elif self.ca_order == "yx":
            temps = [temp_ty, temp_tx]
        elif self.ca_order == "concat":
            temps = [temp_ty]

        self.mca_layers = MCALayers(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    temps=temps,
                                    rate=rate)
        self.token_dropout = tkl.Dropout(rate)

    def call(self, tokens, x, y, mask_x, mask_y, training=None):
        """
        x (B, N, N_patch, d_model) : Input encoded.
        y (B, N, T + 1, d_model) : ground truth encoded.
        tokens (B, N, T, T): task queries.
        mask_x (B, N, 1, T, N_patch): mask of authorization for queries from x.
        mask_y (B, N, 1, T, T + 1): mask of authorization for queries from y.
        """
        if self.ca_order == "xy":
            keys = [x, y]
            masks = [mask_x, mask_y]
        elif self.ca_order == "yx":
            keys = [y, x]
            masks = [mask_y, mask_x]
        elif self.ca_order == "concat":
            keys = [tf.concat([x, y], axis=-2)]
            masks = [tf.concat([mask_x, mask_y], axis=-1)]

        tokens = self.token_dropout(tokens, training=training)
        tokens, blocks = self.mca_layers(query=tokens,
                                         keys=keys,
                                         masks=masks,
                                         training=training)
        return tokens, blocks

class LSALayers(tkl.Layer):
    """ Label Self Attention Layers """
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 temp=1.0,
                 rate=0.1):
        super(LSALayers, self).__init__()
        self.d_model = d_model
        self.lsa_layers = SALayers(num_layers=num_layers,
                                   d_model=d_model,
                                   num_heads=num_heads,
                                   dff=dff,
                                   temp=temp,
                                   rate=rate)

    def call(self, y, mask, training=None):
        """
        y (B, N, T, d_model) : ground truth encoded.
        mask_y (B, N, 1, T, T)
        """
        y, blocks = self.lsa_layers(x=y,
                                    mask=mask,
                                    training=training)
        return y, blocks



