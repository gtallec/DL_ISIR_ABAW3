import tensorflow as tf
import numpy as np
import measures

from metrics_extended.abstract_metrics import SlidingMeanMetric, TensorSlidingMeanMetric

class TreePermutationMetric(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(TreePermutationMetric, self)
         .__init__(name='tree_perm',
                   eval_function=measures.tree_permutation_loss(pred_in)))

class TreePermutationCategoricalMetric(SlidingMeanMetric):
    def __init__(self, pred_in, **kwargs):
        (super(TreePermutationCategoricalMetric, self)
         .__init__(name='tree_perm_cat',
                   eval_function=measures.tree_permutation_categorical_loss(pred_in)))


SUPPORTED_MAONET_METRICS = {"tree_perm": TreePermutationMetric,
                            "tree_categorical": TreePermutationCategoricalMetric}
