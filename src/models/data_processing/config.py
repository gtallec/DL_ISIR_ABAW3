from models.data_processing.data_processing_layers import SUPPORTED_BASIC_DATA_PROCESSING_LAYERS
from models.data_processing.masked_data_processing_layers import SUPPORTED_MASKED_DATA_PROCESSING_LAYERS

SUPPORTED_DATA_PROCESSING_LAYERS = {**SUPPORTED_BASIC_DATA_PROCESSING_LAYERS,
                                    **SUPPORTED_MASKED_DATA_PROCESSING_LAYERS}
