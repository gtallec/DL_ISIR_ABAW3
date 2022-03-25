import copy
import os

import pandas as pd


from datetime import datetime

from configs.config import MACHINE_ID
from configs.experiment_counter import get_experiment_id
from experiment_parser_extended import MetaExperimentParser
from routines_extended import train
from tuning.tuner import fill_meta, get_hyperparam_grid, unroll_constraint

def check_hyperparam_in_df(hyperparams, df):
    cond = not(df.empty)
    hyperparams_keys = list(hyperparams.keys())
    i = 0 
    loc_df = df
    while (i < len(hyperparams_keys)) and cond:
        hyperparam = hyperparams_keys[i]
        if hyperparam in df.columns: 
            hyperval = hyperparams[hyperparam]
            loc_df = loc_df.loc[loc_df[hyperparam] == hyperval]
            cond = not(loc_df.empty)
        i += 1
    return cond

def unroll_hyperparams_name(hyperparams):
    hp_dict = dict({})
    for hyperparam, hyperval in hyperparams.items():
        if isinstance(hyperval, list):
            for i in range(len(hyperval)):
                hp_dict[hyperparam + '_' + str(i)] = hyperval[i]
        else:
            hp_dict[hyperparam] = hyperval
    return hp_dict

def encapsulate_dict(dictionary):
    encapsulated_dict = dict({})
    for key in dictionary:
        encapsulated_dict[key] = [dictionary[key]]
    return encapsulated_dict

def instanciate_hyperparams(experiment_dict, hyperparams, experiment_id, timestamp):
    hyper_experiment_dict = fill_meta(experiment_dict, hyperparams)
    hyper_experiment_dict['log_folder'] = os.path.join(hyper_experiment_dict['log_folder'], experiment_id)
    return hyper_experiment_dict


def store_experiment(experiment_id, timestamp, hyperparams, tracking_path, storing_path):
    meta_dict = {"machine_id": [MACHINE_ID],
                 "experiment_id": [experiment_id],
                 "timestamp": [timestamp],
                 "tracking": [tracking_path]}

    hp_dict = encapsulate_dict(unroll_hyperparams_name(hyperparams))

    df = pd.concat({'meta': pd.DataFrame(meta_dict),
                    'hp': pd.DataFrame(hp_dict)},
                   axis=1)

    if not(os.path.exists(os.path.dirname(storing_path))):
        os.makedirs(os.path.dirname(storing_path))

    if os.path.exists(storing_path):
        df_old = pd.read_csv(storing_path, header=[0, 1])
        df = pd.concat([df_old, df])

    df.to_csv(storing_path, index=False)

def run_hp_and_store_experiment(experiment_dict, hyperparams, constraints=dict(), verbose=False):
    experiment_id = str(get_experiment_id())
    timestamp = datetime.now().strftime("%m-%d-%Y")
    hyper_experiment_dict = instanciate_hyperparams(experiment_dict, 
                                                    unroll_constraint(hyperparams, 
                                                                      constraints),
                                                    experiment_id=experiment_id,
                                                    timestamp=timestamp) 
    tracking_path = train(copy.deepcopy(hyper_experiment_dict), verbose=verbose)
    
    experiment = MetaExperimentParser(hyper_experiment_dict)
    storing_path = experiment.get_storing_path()

    store_experiment(experiment_id=experiment_id,
                     timestamp=timestamp,
                     hyperparams=hyperparams,
                     tracking_path=tracking_path,
                     storing_path=storing_path)

def run_gridsearch(experiment_dict, hyperparams, constraints=dict(), retrain=False, verbose=False): 
    hyperparam_dict_list = get_hyperparam_grid(hyperparams)
    meta_experiment = MetaExperimentParser(experiment_dict)
    storing_path = meta_experiment.get_storing_path()
    for hyperparam_dict in hyperparam_dict_list:
        print('-----------')
        print("HYPERPARAMS")
        for key in hyperparam_dict:
            print(key, hyperparam_dict[key])
        print('-----------', '\n')

        unconstrained_hyperparams = unroll_constraint(hyperparam_dict, constraints=constraints)
        unrolled_hyperparams_name = unroll_hyperparams_name(unconstrained_hyperparams)
        hp_condition = os.path.exists(storing_path)
        run_condition = True 
        if hp_condition:
            hp_in_df = check_hyperparam_in_df(unrolled_hyperparams_name,
                                              pd.read_csv(storing_path, header=[0, 1])['hp'])
            run_condition = (retrain or not(hp_in_df))

        if run_condition:
            run_hp_and_store_experiment(experiment_dict, unconstrained_hyperparams, verbose=verbose)




