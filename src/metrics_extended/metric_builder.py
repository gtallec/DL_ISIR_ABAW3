from metrics_extended.classification_metrics import SUPPORTED_CLASSIFICATION_METRICS
from metrics_extended.loss_metrics import SUPPORTED_LOSS_METRICS
from metrics_extended.threshold_metrics import SUPPORTED_THRESHOLD_METRICS
from metrics_extended.partially_labelled_threshold_metrics import SUPPORTED_PARTIALLY_LABELLED_THRESHOLD_METRICS
from metrics_extended.abstract_metrics import SUPPORTED_ABSTRACT_METRICS

SUPPORTED_METRICS = {**SUPPORTED_LOSS_METRICS,
                     **SUPPORTED_CLASSIFICATION_METRICS,
                     **SUPPORTED_THRESHOLD_METRICS,
                     **SUPPORTED_PARTIALLY_LABELLED_THRESHOLD_METRICS,
                     **SUPPORTED_ABSTRACT_METRICS}

def get_metric(metric_dict, dataset_columns, log_folder):
    metric_type = metric_dict.pop('type')
    return SUPPORTED_METRICS[metric_type](dataset_columns=dataset_columns, log_folder=log_folder, **metric_dict)
