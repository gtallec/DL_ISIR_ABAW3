import tensorflow.keras.models as tkm

from models.data_processing.config import SUPPORTED_DATA_PROCESSING_LAYERS

class DataProcessing(tkm.Model):
    """ Apply successive data processing steps. It is designed to be applied on batches"""
    def __init__(self, data_processing_list, **kwargs):
        super(DataProcessing, self).__init__()
        self.data_processing_layers = []
        for data_processing in data_processing_list:
            self.data_processing_layers.append(get_data_processing_layer(data_processing, **kwargs))

    def call(self, x, y=None, training=None):
        for data_processing_layer in self.data_processing_layers:
            x, y = data_processing_layer(x=x,
                                         training=training,
                                         y=y)
        return x, y

def get_data_processing_layer(data_processing_layer_dict, **kwargs):
    data_processing_type = data_processing_layer_dict.pop('type')
    return SUPPORTED_DATA_PROCESSING_LAYERS[data_processing_type](**data_processing_layer_dict, **kwargs)
