import tensorflow as tf
import pandas as pd
import numpy as np
import os

class DatasetMetric(tf.keras.metrics.Metric):
    def __init__(self, dataset_columns, name, **kwargs):
        print('entering dataset metric constructor')
        super(DatasetMetric, self).__init__(**kwargs)
        self.columns = [name + dataset_column for dataset_column in dataset_columns] + ['mean_' + name]

    def result_to_df(self):
        # (N, )
        result = self.result()
        result_line_with_mean = tf.expand_dims(tf.concat([result, [tf.math.reduce_mean(result)]],
                                                         axis=0),
                                               axis=0).numpy()
        return pd.DataFrame(data=result_line_with_mean, columns=self.columns)

class VectorSlidingMeanMetric(tf.keras.metrics.Metric):
    def __init__(self, vector_in, n_coords, **kwargs):
        (super(VectorSlidingMeanMetric, self)
         .__init__(name=vector_in))
        self.moving_mean = self.add_weight(name=vector_in + '_moving_mean',
                                           shape=(n_coords, ),
                                           initializer='zeros',
                                           dtype=tf.float32)

        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)
        self.vector_in = vector_in
        self.n_coords = n_coords

    def update_state(self, y_true, y_pred, sample_weight=None):
        M = tf.dtypes.cast(tf.shape(y_true)[0], dtype=tf.float32)
        combination_coeff = self.N / (self.N + M)
        vector = y_pred[self.vector_in]
        self.moving_mean.assign(self.moving_mean * combination_coeff + (1 - combination_coeff) * tf.reshape(vector, (self.n_coords, )))
        self.N.assign_add(M)

    def result(self):
        return self.moving_mean

    def result_to_df(self):
        result_columns = [self.vector_in + '_' + str(i) for i in range(self.n_coords)]
        result = self.result().numpy().reshape(1, self.n_coords)
        return pd.DataFrame(data=result, columns=result_columns)

    def reset_states(self):
        self.moving_mean.assign(tf.zeros_like(self.moving_mean))

class MatrixSlidingMeanMetric(tf.keras.metrics.Metric):
    def __init__(self, matrix_in, shape, **kwargs):
        (super(MatrixSlidingMeanMetric, self)
         .__init__(name=matrix_in))

        self.moving_mean = self.add_weight(name=matrix_in + '_' + 'moving_mean',
                                           shape=shape,
                                           initializer='zeros',
                                           dtype=tf.float32)
        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)
        self.matrix_in = matrix_in
        self.shape = shape


    def update_state(self, y_true, y_pred, sample_weight=None):
        M = tf.dtypes.cast(tf.shape(y_true)[0], dtype=tf.float32)
        combination_coeff = self.N / (self.N + M)
        matrix = tf.math.reduce_mean(y_pred[self.matrix_in], axis=0)
        self.moving_mean.assign(self.moving_mean * combination_coeff + (1 - combination_coeff) * matrix)
        self.N.assign_add(M)

    def result(self):
        return self.moving_mean

    def result_to_df(self):
        result_columns = []
        result_in_line = []
        results = self.result().numpy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result_in_line.append(results[i, j])
                result_columns.append(self.matrix_in + '_' + str(i) + str(j))
        return pd.DataFrame(data=result_in_line, columns=result_columns)

    def reset_states(self):
        self.moving_mean.assign(tf.zeros_like(self.moving_mean))
        



class ScalarTrackingMetric(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super(ScalarTrackingMetric, self).__init__()
        self.column_name = name

    def result_to_df(self):
        result = self.result().numpy()
        # tf.print(self.column_name, ': ', result)
        return pd.DataFrame(data=[self.result().numpy()], columns=[self.column_name])

class SlidingMeanMetric(ScalarTrackingMetric):
    def __init__(self, name, eval_function, **kwargs):
        (super(SlidingMeanMetric, self)
         .__init__(name=name,
                   **kwargs))
        self.eval_function = eval_function
        self.moving_mean = self.add_weight(name=name + '_moving_mean',
                                           initializer='zeros',
                                           dtype=tf.float32)

        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        M = tf.dtypes.cast(tf.shape(y_true)[0], dtype=tf.float32)
        combination_coeff = self.N / (self.N + M)
        batch_eval = self.eval_function(y_true, y_pred)
        self.moving_mean.assign(self.moving_mean * combination_coeff + (1 - combination_coeff) * batch_eval)
        self.N.assign_add(M)

    def result(self):
        return self.moving_mean

class TensorTrackingMetric(tf.keras.metrics.Metric):
    def __init__(self, shape, tensor_in, log_folder, **kwargs):
        super(TensorTrackingMetric, self).__init__()
        self.end_value = self.add_weight(name=tensor_in,
                                         initializer='zeros',
                                         shape=shape,
                                         dtype=tf.float32)
        self.shape = shape
        self.log_folder = log_folder
        self.tensor_in = tensor_in
        self.n_res = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[self.tensor_in]
        self.end_value.assign(y_pred)

    def result(self):
        return self.end_value

    def result_to_df(self):
        tensor_result = self.result().numpy()
        result_file = os.path.join(self.log_folder, "{}_{}.npy".format(self.tensor_in, self.n_res))
        np.save(result_file, tensor_result)
        result_df = pd.DataFrame(data=[result_file], columns=[self.tensor_in])
        self.n_res += 1
        return result_df
     
    def reset_states(self):
        self.end_value.assign(tf.zeros_like(self.end_value))


class ParameterTrackingMetric(ScalarTrackingMetric):
    def __init__(self, param_in, eval_function, name, **kwargs):
        super(ParameterTrackingMetric, self).__init__(name=name + '_' + param_in)
        self.param_in = param_in
        self.eval_function = eval_function
        self.end_value = self.add_weight(name=name + '_parameter',
                                         initializer='zeros',
                                         dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = y_pred[self.param_in]
        self.end_value.assign(self.eval_function(y_pred))

    def result(self):
        return self.end_value

class TensorSlidingMeanMetric(tf.keras.metrics.Metric):
    def __init__(self, name, eval_function, shape, **kwargs):
        (super(TensorSlidingMeanMetric, self)
         .__init__(**kwargs))
        self.eval_function = eval_function
        self.shape = shape
        self.moving_mean = self.add_weight(name=name + "_moving_mean",
                                           shape=shape,
                                           initializer='zeros',
                                           dtype=tf.float32)
        self.N = self.add_weight(name='N',
                                 initializer='zeros',
                                 dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        M = tf.dtypes.cast(tf.shape(y_true)[0], dtype=tf.float32)
        combination_coeff = self.N / (self.N + M)
        batch_eval = self.eval_function(y_true, y_pred)
        self.moving_mean.assign(self.moving_mean * combination_coeff + (1 - combination_coeff) * batch_eval)
        self.N.assign_add(M)

    def result(self):
        return self.moving_mean

    def reset_states(self):
        self.moving_mean.assign(tf.zeros(self.shape))
        self.N.assign(0)


SUPPORTED_ABSTRACT_METRICS = {"vtracking": VectorSlidingMeanMetric}

