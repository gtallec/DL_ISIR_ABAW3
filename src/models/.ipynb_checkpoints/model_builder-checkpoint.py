from models.regressors.config import SUPPORTED_REGRESSORS
from models.encoders.config import SUPPORTED_ENCODERS
from models.vectors import SUPPORTED_VECTORS
from models.skeletons import SUPPORTED_SKELETONS

SUPPORTED_MODELS = {**SUPPORTED_REGRESSORS,
                    **SUPPORTED_ENCODERS,
                    **SUPPORTED_VECTORS,
                    **SUPPORTED_SKELETONS}

def model_block(model_args, name):
    model_type = model_args.pop('type')
    return SUPPORTED_MODELS[model_type](**model_args, name=name) 

class ModelBuilder:
    def __init__(self, model_dict):
        self.model_dict = model_dict
        self.dependency_checker = []
        self.built_model = dict()

    def build_model(self, model_name):
        model_dict = self.model_dict[model_name]
        dependencies = model_dict.pop('dependencies', [])
        optimizer = model_dict.pop('optimizer', None)
        pretrained_weights = model_dict.pop('pretrained_weights', None)

        for dependency in dependencies:
            if not(dependency in self.dependency_checker):
                self.build_model(dependency)
            model_dict[dependency] = self.built_model[dependency]['instance']
        model_instance = model_block(model_args=model_dict,
                                     name=model_name)
        
        self.built_model[model_name] = dict()
        if pretrained_weights is not None:
            self.built_model[model_name]['pretrained_weights'] = pretrained_weights

        self.built_model[model_name]['instance'] = model_instance
        self.built_model[model_name]['dependencies'] = dependencies
        if optimizer is not None:
            self.built_model[model_name]['optimizer'] = optimizer

    def parse_model(self):
        self.build_model('main')
        self.model = self.built_model['main']['instance']
        return self.model, self.built_model
