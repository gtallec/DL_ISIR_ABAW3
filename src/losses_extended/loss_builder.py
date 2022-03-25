import tensorflow.keras.losses as tkl

from losses_extended.permutation_losses import SUPPORTED_PERMUTATION_LOSSES
from losses_extended.classification_losses import SUPPORTED_CLASSIFICATION_LOSSES
from losses_extended.regularization import SUPPORTED_REGULARIZATION
from losses_extended.reweighted_losses import SUPPORTED_REWEIGHTED_LOSSES

SUPPORTED_LOSSES = {**SUPPORTED_PERMUTATION_LOSSES,
                    **SUPPORTED_CLASSIFICATION_LOSSES,
                    **SUPPORTED_REGULARIZATION,
                    **SUPPORTED_REWEIGHTED_LOSSES,
                    "cce": tkl.CategoricalCrossentropy,
                    "mse": tkl.MeanSquaredError,
                    "mae": tkl.MeanAbsoluteError}

def get_loss(loss_dict):
    loss_type = loss_dict.pop('type')
    # print('loss type')
    return SUPPORTED_LOSSES[loss_type](**loss_dict) 
