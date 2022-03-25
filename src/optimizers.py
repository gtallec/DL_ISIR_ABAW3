import tensorflow_addons as tfa
import tensorflow.keras.optimizers as tko

"""Supported Optimizers"""
OPTIMIZERS = {'adadelta': tko.Adadelta,
              'adagrad': tko.Adagrad,
              'adam': tko.Adamax,
              'adamw': tfa.optimizers.AdamW,
              'ftrl': tko.Ftrl,
              'nadam': tko.Nadam,
              'rmsprop': tko.RMSprop,
              'sgd': tko.SGD}

def optimizer(optimizer_args):
    optimizer_type = optimizer_args.pop('type')
    return OPTIMIZERS[optimizer_type](**optimizer_args)


if __name__ == '__main__':
    adamW = optimizer({"type": "adamw",
                       "weight_decay": 0.5})
