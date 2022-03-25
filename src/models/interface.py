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
        opt_dict = dict()
        trainable_variables = dict()
        for built_model in self.built_models:
            if 'optimizer' in self.built_models[built_model]:
                opt_dict[built_model] = self.built_models[built_model]['optimizer']
                net = self.built_models[built_model]['instance']
                dependent_net_trainable_variables = net.trainable_variables

                net_trainable_variables = []
                dependencies = list(self.built_models[built_model]['dependencies'])
                for var in dependent_net_trainable_variables:
                    has_dependency = any([dependency in var.name for dependency in dependencies])
                    if not has_dependency:
                        net_trainable_variables.append(var)
                """
                print(50 * "#")
                print(built_model.upper())
                for var in net_trainable_variables:
                    print(var.name)
                print(50 * "#")
                """

                trainable_variables[built_model] = net_trainable_variables
        return opt_dict, trainable_variables 


    def get_trainable_variables_deprecated(self):
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
                pretrained_weights_path = self.built_models[built_model]['pretrained_weights']
                weights_path = os.path.join(PRETRAINED_WEIGHTS_PATH, pretrained_weights_path)
                print('LOADING WEIGHTS FROM {}'.format(weights_path))
                self.load_weights(weights_path,
                                  built_model)

    def save_weights(self, ckpt_path, block='main', **kwargs):
        if block == "main":
            super(DictionaryModel, self).save_weights(ckpt_path, **kwargs)
        elif block in self.built_models:
            self.built_models[block]['instance'].save_weights(ckpt_path, **kwargs)
        else:
            pass

    def load_weights(self, ckpt_path, block):
        """Load weights from ckpt_path into the part of the net specified by name block"""
        if block == "main":
            # print('Loading full weights') 
            super(DictionaryModel, self).load_weights(ckpt_path)

        elif block in self.built_models:
            # print('Loading weights for {}'.format(block))
            self.built_models[block]['instance'].load_weights(ckpt_path)

        else:
            pass
            print('Block {} does not exist')

def get_model(model_dict, **kwargs):
    return DictionaryModel(*(ModelBuilder(model_dict, **kwargs).parse_model()))

