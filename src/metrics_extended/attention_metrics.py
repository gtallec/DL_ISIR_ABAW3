import matplotlib.pyplot as plt
import os
# import cv2
import measures

import numpy as np
import pandas as pd
import tensorflow as tf

from metrics_extended.abstract_metrics import TensorSlidingMeanMetric

"""
class AttentionMapOnImage(tf.keras.metrics.Metric):
    def __init__(self, image_in, attention_in, image_shape, attention_shape, log_folder, name='attention_map', **kwargs):
        # image_in (str) : output dict key for the image,
        # attention_in (str): output dict key for the attention map,
        # image_shape (N_i, M_i, 3) : image shape,
        # attention_shape (N_a, M_a): attention map shape

        super(AttentionMapOnImage, self).__init__(name=name)
        self.image_in = image_in
        self.attention_in = attention_in
        self.log_folder = log_folder
        self.save_name = name
        self.n_res = 0

        self.image_shape = image_shape
        self.attention_shape = attention_shape

        self.image_with_attention = self.add_weight(name='image',
                                                    initializer='zeros',
                                                    shape=image_shape)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # (N_i, M_i, 3) 
        image = y_pred[self.image_in][0]
        # (N_a, M_a)
        attention_map = tf.reshape(y_pred[self.attention_in][0], self.attention_shape)
        attention_map = tf.tile(tf.expand_dims(attention_map, axis=-1),
                                (1, 1, 3))
        attention_map = tf.image.resize(attention_map, self.image_shape[:-1])
        image_with_attention = image + attention_map
        self.image_with_attention.assign(255 * image_with_attention)

    def reset_states(self):
        self.image_with_attention.assign(tf.zeros(self.image_shape))

    def result(self):
        return self.image_with_attention

    def result_to_df(self):
        image_with_attention = self.image_with_attention.numpy()
        result_columns = [self.save_name]
        image_file = os.path.join(self.log_folder, '{}_{}.jpg'.format(self.name, self.n_res))
        cv2.imwrite(image_file, image_with_attention[:, :, ::-1])
        result_df = pd.DataFrame(data=[image_file], columns=result_columns)
        self.n_res += 1
        return result_df
"""

class AttentionMap(tf.keras.metrics.Metric):
    def __init__(self, attention_in, attention_shape, log_folder, **kwargs):
        """
        attention_in (str): output dict key for the attention map,
        attention_shape (N_a, M_a): attention map shape
        """
        super(AttentionMap, self).__init__(name=attention_in)
        self.attention_in = attention_in
        self.log_folder = log_folder
        self.save_name = attention_in 
        self.n_res = 0

        self.attention_shape = attention_shape

        self.attention_map = self.add_weight(name='attention',
                                             initializer='zeros',
                                             shape=attention_shape)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # (N_a, M_a)
        attention_map = tf.reshape(y_pred[self.attention_in], self.attention_shape)
        self.attention_map.assign(attention_map)

    def reset_states(self):
        self.attention_map.assign(tf.zeros(self.attention_shape))

    def result(self):
        return self.attention_map

    def result_to_df(self):
        attention_map = self.attention_map.numpy()
        result_columns = [self.save_name]
        attention_file = os.path.join(self.log_folder, '{}_{}.npy'.format(self.name, self.n_res))
        np.save(attention_file, attention_map)
        result_df = pd.DataFrame(data=[attention_file], columns=result_columns)
        self.n_res += 1
        return result_df

class MeanAttentionMap(TensorSlidingMeanMetric):
    def __init__(self, attention_in, attention_shape, num_heads, log_folder, **kwargs):
        """
        image_in (str) : output dict key for the image,
        attention_in (str): output dict key for the attention map,
        attention_shape (N_a, M_a): attention map shape
        "num_heads": Number of attention heads.
        """
        super(MeanAttentionMap, self).__init__(name=attention_in,
                                               shape=(num_heads, *attention_shape),
                                               eval_function=measures.mean_tensor(attention_in))

        self.attention_shape = (num_heads, *attention_shape)
        self.attention_in = attention_in
        self.log_folder = log_folder
        if not(os.path.exists(self.log_folder)):
            os.makedirs(self.log_folder)
        self.save_name = attention_in
        self.n_res = 0


    def result_to_df(self):
        attention_map = self.result().numpy()
        result_columns = [self.save_name]
        attention_file = os.path.join(self.log_folder, '{}_{}.npy'.format(self.name, self.n_res))
        np.save(attention_file, attention_map)
        result_df = pd.DataFrame(data=[attention_file], columns=result_columns)
        self.n_res += 1
        return result_df

SUPPORTED_ATTENTION_METRICS = {"att": AttentionMap,
                               "mean_att": MeanAttentionMap}








