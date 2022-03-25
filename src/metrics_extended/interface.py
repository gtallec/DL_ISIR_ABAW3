import tensorflow as tf
import pandas as pd
import copy as cp

from metrics_extended.metric_builder import get_metric

class MultiOutputsMetrics(tf.keras.metrics.Metric):
    """ 
    Apply different metrics to different outputs, multiple metrics to the same output is supported,
    however metric names should be unique.
    """
    def __init__(self, metric_kwargs_list, dataset_columns=None, log_folder=None, **kwargs):
        super(MultiOutputsMetrics, self).__init__(**kwargs)
        self.metric_list = []
        self.metric_kwargs_list = cp.deepcopy(metric_kwargs_list) 
        for metric_kwargs in metric_kwargs_list:
            self.metric_list.append(get_metric(metric_kwargs, dataset_columns, log_folder))

    def reset_states(self):
        for metric in self.metric_list:
            metric.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        for metric in self.metric_list:
            metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return [metric.result() for metric in self.metric_list]

    def result_to_df(self):
        if len(self.metric_list) == 0:
            return pd.DataFrame()
        return pd.concat([metric.result_to_df() for metric in self.metric_list], axis=1)

    def get_metric(self, metric_type):
        found = False
        i = 0
        while not(found) and (i < len(self.metric_list)):
            metric_kwargs = self.metric_kwargs_list[i]
            metric = self.metric_list[i]
            found = (metric_kwargs['type'] == metric_type)
            i += 1
        return found, metric

def get_metrics(metrics_dict, dataset_columns=None, log_folder=None):
    return MultiOutputsMetrics(metrics_dict, dataset_columns, log_folder)
