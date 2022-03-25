import datasets.abaw3.config as abaw_conf
import datasets.bp4d.config as bp4d_conf
import datasets.disfa.config as disfa_conf

import os
import numpy as np

ABAW_ORDER = abaw_conf.AU_ORDER
BP4D_ORDER = bp4d_conf.AU_ORDER
DISFA_ORDER = ['AU{}'.format(au) for au in disfa_conf.AU_ORDER]
AU_ORDER = sorted(list(set(ABAW_ORDER) | set(BP4D_ORDER) | set(DISFA_ORDER)), key=lambda x: int(x[2:]))


def map_src_to_target(src_order, target_order):
    mapping = len(src_order) * np.ones((len(target_order))).astype(int)
    for i in range(len(target_order)):
        for j in range(len(src_order)):
            if src_order[j] == target_order[i]:
                mapping[i] = j
    return list(mapping)

ABAW_TO_ABS = map_src_to_target(ABAW_ORDER, AU_ORDER)
BP4D_TO_ABS = map_src_to_target(BP4D_ORDER, AU_ORDER)
DISFA_TO_ABS = map_src_to_target(DISFA_ORDER, AU_ORDER)

BP4D_DISFA_CSV_TEMPLATE = os.path.join('..', 'resources', 'ABAW3', 'preprocessed', '{}_disfa_bp4d.csv')

print(ABAW_TO_ABS)
