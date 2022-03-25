import tensorflow as tf
import measures
from losses_extended.abstract_losses import MeasureLoss

class L2Regularization(MeasureLoss):
    def __init__(self, pred_in):
        super(L2Regularization, self).__init__(measures.l2(pred_in=pred_in))

SUPPORTED_REGULARIZATION = {"l2": L2Regularization}
