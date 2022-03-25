from datasets.disfa.config import AU_ORDER, get_projection

import numpy as np
def columns_disfa(projection_mode='sota', **kwargs):
    projection = get_projection(projection_mode)
    order = np.array(AU_ORDER)[projection]
    return ['AU{}'.format(au) for au in order]

def columns_disfa_for_dysfer(mix12=False, **kwargs):
    if mix12:
        return ['AU1&2', 'AU4', 'AU5']
    else:
        return ['AU1', 'AU2', 'AU4', 'AU5']
