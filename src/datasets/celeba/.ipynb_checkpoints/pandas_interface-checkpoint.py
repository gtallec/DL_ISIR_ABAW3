from datasets.celeba.config import LABELS_NAME, get_projection 
import numpy as np

def columns_celeba(projection_mode, **kwargs):
    projection = get_projection(projection_mode)
    return list(np.array(LABELS_NAME)[projection])

