import tensorflow as tf
import numpy as np
import measures

from metrics_extended.abstract_metrics import SlidingMeanMetric, TensorSlidingMeanMetric, TensorTrackingMetric

class ModuleWeightsTracking(TensorTrackingMetric):
    def __init__(self, D, T, M, sigma_in, log_folder, **kwargs):
        super(ModuleWeightsTracking, self).__init__(shape=(D, T, M),
                                                    tensor_in=sigma_in,
                                                    log_folder=log_folder)

SUPPORTED_MEYERSON_METRICS = {"modw_tracking": ModuleWeightsTracking}
