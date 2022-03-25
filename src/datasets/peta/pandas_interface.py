from datasets.peta.config import PETA_PATH, get_projection
import pandas as pd
import os
import numpy as np

def columns_peta(projection_mode, **kwargs):
    projection = get_projection(projection_mode)
    return list(np.array(list(pd.read_csv(os.path.join(PETA_PATH, 'interfaces', 'partition_0', 'test.csv'))
                              .drop('path', axis=1)
                              .keys()))[projection])
