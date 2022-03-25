import argparse
import os

def file_path(string):
    if os.path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()   
parser.add_argument('-g', '--hypergrid', required=True, default='sanity')
parser.add_argument('-e', '--experiment', required=True)
parser.add_argument('-d', '--device', required=False, default=None)


args = parser.parse_args()

if not(args.device is None):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

hypergrid = args.hypergrid
experiment = args.experiment 

import json
import tuning_routines
import tensorflow as tf
import numpy as np
from utils import dict_from_json


hp_path = file_path(hypergrid) 
grid_and_constraints = dict_from_json(hp_path)
hyperparams = grid_and_constraints.get('grid', {})
constraints = grid_and_constraints.get('constraints', {})

experiment = file_path(experiment) 
experiment_dict = dict_from_json(experiment)
tuning_routines.run_gridsearch(experiment_dict,
                               hyperparams=hyperparams,
                               constraints=constraints,
                               verbose=False,
                               retrain=True)
