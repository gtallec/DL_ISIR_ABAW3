import os
PETA_PATH = os.path.join('..', 'resources', 'PETA')


SUPPORTED_PROJECTIONS = {"all": [i for i in range(40)],
                         "sota": [i for i in range(35)]}

def get_projection(projection_mode):
    return SUPPORTED_PROJECTIONS[projection_mode]
