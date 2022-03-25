from measures_extended.monet_measures import frobnorm_to_mat
from metrics_extended.abstract_metrics import SlidingMeanMetric

import tensorflow as tf

class Frob2Mat(SlidingMeanMetric):
    def __init__(self, soft_order, mat, name='frob2mat', **kwargs):
        (super(Frob2Mat, self)
         .__init__(name=name,
                   eval_function=frobnorm_to_mat(soft_order=soft_order,
                                                 mat=mat)))

class Frob2Id(Frob2Mat):
    def __init__(self, soft_order, T, name='frob2id', **kwargs):
        (super(Frob2Id, self)
         .__init__(name=name,
                   soft_order=soft_order,
                   mat=tf.eye(T),
                   **kwargs))

SUPPORTED_MONET_METRICS = {"frob2Id": Frob2Id}
                   

