from metrics_extended.classification_metrics import SUPPORTED_CLASSIFICATION_METRICS
from metrics_extended.categorical_classification_metrics import SUPPORTED_CATEGORICAL_CLASSIFICATION_METRICS
from metrics_extended.utils_metrics import SUPPORTED_UTILS_METRICS
from metrics_extended.recurrent_metrics import SUPPORTED_RECURRENT_METRICS
from metrics_extended.loss_metrics import SUPPORTED_LOSS_METRICS
from metrics_extended.permutation_metrics import SUPPORTED_PERMUTATION_METRICS
from metrics_extended.maonet_metrics import SUPPORTED_MAONET_METRICS
from metrics_extended.stat_metrics import SUPPORTED_STAT_METRICS
from metrics_extended.threshold_metrics import SUPPORTED_THRESHOLD_METRICS
from metrics_extended.matrix_metrics import SUPPORTED_MATRIX_METRICS
from metrics_extended.parameter_trackers import SUPPORTED_PARAMETER_TRACKERS


SUPPORTED_METRICS = {**SUPPORTED_PERMUTATION_METRICS,
                     **SUPPORTED_UTILS_METRICS,
                     **SUPPORTED_RECURRENT_METRICS,
                     **SUPPORTED_LOSS_METRICS,
                     **SUPPORTED_PERMUTATION_METRICS,
                     **SUPPORTED_CLASSIFICATION_METRICS,
                     **SUPPORTED_MAONET_METRICS,
                     **SUPPORTED_STAT_METRICS,
                     **SUPPORTED_THRESHOLD_METRICS,
                     **SUPPORTED_PARAMETER_TRACKERS,
                     **SUPPORTED_CATEGORICAL_CLASSIFICATION_METRICS,
                     **SUPPORTED_MATRIX_METRICS}

def get_metric(metric_dict, dataset_columns, log_folder):
    metric_type = metric_dict.pop('type')
    print("metric_dict : ", metric_dict)
    return SUPPORTED_METRICS[metric_type](dataset_columns=dataset_columns, log_folder=log_folder, **metric_dict)
