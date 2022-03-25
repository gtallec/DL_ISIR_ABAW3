from models.model_builder import ModelBuilder
import tensorflow.keras.models as tkm
import tensorflow as tf
from configs.config import PRETRAINED_WEIGHTS_PATH
import os

class DictionaryModel(tkm.Model):
    def __init__(self, model_instance, built_models, **kwargs):
        super(DictionaryModel, self).__init__(**kwargs)
        self.model = model_instance
        self.built_models = built_models

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)

    def get_trainable_variables(self):
        trainable_variables = []
        block_dict = dict()
        cursor = 0

        for built_model in self.built_models:
            if 'optimizer' in self.built_models[built_model]:
                block_dict[built_model] = dict()
                block_dict[built_model]['start'] = cursor
                net = self.built_models[built_model]['instance']
                dependent_net_trainable_variables = net.trainable_variables


                net_trainable_variables = []
                dependencies = list(self.built_models[built_model]['dependencies'])

                for var in dependent_net_trainable_variables:
                    has_dependency = any([dependency in var.name for dependency in dependencies])
                    if not has_dependency:
                        net_trainable_variables.append(var)

                """
                for variable in net_trainable_variables:
                    tf.print(variable.name)
                """

                block_dict[built_model]['variables'] = net_trainable_variables

                cursor += len(net_trainable_variables)
                block_dict[built_model]['stop'] = cursor
                block_dict[built_model]['optimizer'] = self.built_models[built_model]['optimizer']
                trainable_variables += net_trainable_variables
        
        return block_dict, trainable_variables

    def load_pretrained_weights(self):
        for built_model in self.built_models:
            if 'pretrained_weights' in self.built_models[built_model]:
                print('LOADING PRETRAINED WEIGHTS : ')
                pretrained_weights_path = self.built_models[built_model]['pretrained_weights']
                self.load_weights(os.path.join(PRETRAINED_WEIGHTS_PATH, pretrained_weights_path),
                                  built_model)

    def load_weights(self, ckpt_path, block):
        """Load weights from ckpt_path into the part of the net specified by name block"""

        if block == "main":
            print('Loading full weights') 
            super(DictionaryModel, self).load_weights(ckpt_path)

        elif block in self.built_models:
            print('Loading weights for {}'.format(block))
            self.built_models[block]['instance'].load_weights(ckpt_path)

        else:
            print('Block {} does not exist')

def get_model(model_dict):
    return DictionaryModel(*(ModelBuilder(model_dict).parse_model()))

