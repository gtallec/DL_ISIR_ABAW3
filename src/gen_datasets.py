from datasets.toys.generation import gen_toy
from datasets.toys.visualisation import visu_toy
from datasets.toys.pandas_interface import columns_toy

from datasets.celeba.generation import gen_celeba
from datasets.celeba.visualisation import visu_celeba
from datasets.celeba.pandas_interface import columns_celeba

from datasets.disfa.generation import gen_disfa_v2
from datasets.disfa.visualisation import visu_disfa
from datasets.disfa.pandas_interface import columns_disfa

from datasets.disfa.generation import gen_disfa_for_dysfer
from datasets.disfa.pandas_interface import columns_disfa_for_dysfer

from datasets.dysfer.generation import gen_dysfer
from datasets.dysfer.visualisation import visu_dysfer
from datasets.dysfer.pandas_interface import columns_dysfer

from datasets.abaw2.generation import gen_abaw2
from datasets.abaw2.pandas_interface import columns_abaw2

from datasets.abaw3.generation import gen_abaw3
from datasets.abaw3.video_generation import gen_abaw3_video
from datasets.abaw3.pandas_interface import columns_abaw3

from datasets.nuswide.generation import gen_nuswide
from datasets.nuswide.pandas_interface import columns_nuswide

from datasets.jaad.generation import gen_jaad
from datasets.jaad.pandas_interface import columns_jaad

from datasets.peta.generation import gen_peta
from datasets.peta.pandas_interface import columns_peta

from datasets.bp4d.generation import gen_bp4d
from datasets.bp4d.pandas_interface import columns_bp4d

from datasets.au_merge.generation import gen_au_merge
from datasets.au_merge.pandas_interface import columns_au_merge

from datasets.au_merge.generation import gen_bp4d_disfa

import copy
import mappers


DATASET_GENERATION_MAPPING = {'disfa': gen_disfa_v2,
                              'disfa_for_dysfer': gen_disfa_for_dysfer,
                              'dysfer': gen_dysfer, 
                              'toy': gen_toy,
                              'celeba': gen_celeba,
                              'jaad': gen_jaad,
                              'abaw2': gen_abaw2,
                              'abaw3': gen_abaw3,
                              'abaw3_video': gen_abaw3_video,
                              'nuswide': gen_nuswide,
                              'peta': gen_peta,
                              'bp4d': gen_bp4d,
                              'au_merge': gen_au_merge,
                              'bp4d_disfa': gen_bp4d_disfa}

DATASET_VISUALISATION_MAPPING = {'disfa': visu_disfa,
                                 'dysfer': visu_dysfer,
                                 'toy': visu_toy,
                                 'celeba': visu_celeba,
                                 'subdisfa': visu_disfa}

DATASET_DF_COLUMNS = {"disfa": columns_disfa,
                      "dysfer": columns_dysfer,
                      "disfa_for_dysfer": columns_disfa_for_dysfer,
                      "toy": columns_toy,
                      "celeba": columns_celeba,
                      "jaad": columns_jaad,
                      "abaw2": columns_abaw2,
                      "abaw3": columns_abaw3,
                      "abaw3_video": columns_abaw3,
                      "nuswide": columns_nuswide,
                      "au_merge": columns_au_merge,
                      "peta": columns_peta,
                      "bp4d": columns_bp4d,
                      "bp4d_disfa": columns_au_merge}

def gen_dataset(dataset, **kwargs):
    dataset_copy = copy.deepcopy(dataset)
    name = dataset_copy.pop('name')
    dataset = DATASET_GENERATION_MAPPING[name](**dataset_copy)
    return dataset

def df_columns_dataset(storing_meta, **kwargs):
    storing_meta_copy = copy.deepcopy(storing_meta)
    name = storing_meta_copy.pop('name')
    return DATASET_DF_COLUMNS[name](**storing_meta_copy, **kwargs)

def visu_dataset(dataset, **kwargs): 
    dataset_copy = copy.deepcopy(dataset)
    name = dataset_copy.pop('name')
    return DATASET_VISUALISATION_MAPPING[name](**dataset_copy, **kwargs)


if __name__ == '__main__':
    import tensorflow as tf
    dataset = {"name": "nuswide",
               "mode": "train",
               "batchsize": 32}
    nuswide = gen_dataset(dataset).take(10)
    for (images, labels) in nuswide:
        tf.print(tf.math.reduce_min(images, axis=(1, 2, 3)))
        tf.print(tf.math.reduce_max(images, axis=(1, 2, 3)))
