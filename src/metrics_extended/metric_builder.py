from metrics_extended.classification_metrics import SUPPORTED_CLASSIFICATION_METRICS
from metrics_extended.masked_classification_metrics import SUPPORTED_MASKED_CLASSIFICATION_METRICS
from metrics_extended.categorical_classification_metrics import SUPPORTED_CATEGORICAL_CLASSIFICATION_METRICS
from metrics_extended.utils_metrics import SUPPORTED_UTILS_METRICS
from metrics_extended.recurrent_metrics import SUPPORTED_RECURRENT_METRICS
from metrics_extended.loss_metrics import SUPPORTED_LOSS_METRICS
from metrics_extended.permutation_metrics import SUPPORTED_PERMUTATION_METRICS
from metrics_extended.maonet_metrics import SUPPORTED_MAONET_METRICS
from metrics_extended.stat_metrics import SUPPORTED_STAT_METRICS
from metrics_extended.threshold_metrics import SUPPORTED_THRESHOLD_METRICS
from metrics_extended.masked_threshold_metrics import SUPPORTED_MASKED_THRESHOLD_METRICS
from metrics_extended.partially_labelled_threshold_metrics import SUPPORTED_PARTIALLY_LABELLED_THRESHOLD_METRICS
from metrics_extended.matrix_metrics import SUPPORTED_MATRIX_METRICS
from metrics_extended.parameter_trackers import SUPPORTED_PARAMETER_TRACKERS
from metrics_extended.meyerson_metrics import SUPPORTED_MEYERSON_METRICS
from metrics_extended.attention_metrics import SUPPORTED_ATTENTION_METRICS
from metrics_extended.monet_metrics import SUPPORTED_MONET_METRICS
from metrics_extended.abstract_metrics import SUPPORTED_ABSTRACT_METRICS
from metrics_extended.reweighted_metrics import SUPPORTED_REWEIGHTED_METRICS

SUPPORTED_METRICS = {**SUPPORTED_PERMUTATION_METRICS,
                     **SUPPORTED_UTILS_METRICS,
                     **SUPPORTED_RECURRENT_METRICS,
                     **SUPPORTED_LOSS_METRICS,
                     **SUPPORTED_PERMUTATION_METRICS,
                     **SUPPORTED_CLASSIFICATION_METRICS,
                     **SUPPORTED_MASKED_CLASSIFICATION_METRICS,
                     **SUPPORTED_MAONET_METRICS,
                     **SUPPORTED_STAT_METRICS,
                     **SUPPORTED_THRESHOLD_METRICS,
                     **SUPPORTED_MASKED_THRESHOLD_METRICS,
                     **SUPPORTED_PARAMETER_TRACKERS,
                     **SUPPORTED_CATEGORICAL_CLASSIFICATION_METRICS,
                     **SUPPORTED_PARTIALLY_LABELLED_THRESHOLD_METRICS,
                     **SUPPORTED_MEYERSON_METRICS,
                     **SUPPORTED_MATRIX_METRICS,
                     **SUPPORTED_ABSTRACT_METRICS,
                     **SUPPORTED_ATTENTION_METRICS,
                     **SUPPORTED_MONET_METRICS,
                     **SUPPORTED_REWEIGHTED_METRICS}

def get_metric(metric_dict, dataset_columns, log_folder):
    metric_type = metric_dict.pop('type')
    return SUPPORTED_METRICS[metric_type](dataset_columns=dataset_columns, log_folder=log_folder, **metric_dict)
