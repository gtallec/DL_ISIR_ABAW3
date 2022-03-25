from callbacks.checkpoint_callbacks import SUPPORTED_CHECKPOINT_CALLBACKS
from callbacks.schedulers import SUPPORTED_SCHEDULERS
from callbacks.permutation_callbacks import SUPPORTED_PERMUTATION_CALLBACKS

SUPPORTED_CALLBACKS = {**SUPPORTED_SCHEDULERS,
                       **SUPPORTED_CHECKPOINT_CALLBACKS}

def get_callback(callback_args):
    callback_type = callback_args.pop('type')
    return SUPPORTED_CALLBACKS[callback_type](**callback_args)
