import tensorflow as tf
import pandas as pd


class DatasetMetric(tf.keras.metrics.Metric):
    def __init__(self, dataset_columns, name, **kwargs):
        super(DatasetMetric, self).__init__(**kwargs)
        self.columns = [name + dataset_column for dataset_column in dataset_columns] + ['mean_' + name]

    def result_to_df(self):
        # (N, )
        result = self.result()
        result_line_with_mean = tf.expand_dims(tf.concat([result, [tf.math.reduce_mean(result)]],
                                                         axis=0),
                                               axis=0).numpy()
        return pd.DataFrame(data=result_line_with_mean, columns=self.columns)

class ScalarTrackingMetric(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super(ScalarTrackingMetric, self).__init__()
        self.column_name = name

    def result_to_df(self):
        result = self.result().numpy()
        tf.print(self.column_name, ': ', result)
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


class ParameterTrackingMetric(ScalarTrackingMetric):
    def __init__(self, param_in, eval_function, name, **kwargs):
        print("name : ", name)
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
