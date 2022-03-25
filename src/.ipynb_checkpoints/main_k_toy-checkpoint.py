import os
import json

import routines_extended

hyperparams = {"batchsize": [32, 64],
               "encoder_bottleneck": [128],
               'mixture_units': [16, 32],
               'recurrent_units': [32, 64],
               'dropout': [16, 24],
               "N_sample": [10],
               "monet_init": [False, True],
               "lr0_decay_shared_lr": [0.99],
               "lr0_shared_lr": [5e-4],
               "lr0_permutation": [0.0],
               "lr0_threshold_permutation": [10, 50],
               "lr1_permutation": [5e-3, 1e-2]}

constraints = {'shared_lr': ['encoder', 'regressor'],
               'shared_units': ['mixture_units', 'recurrent_units']}
# hyperparam_grid(hyperparams)


train_experiment_template = os.path.join('..', 'experiments', 'CVPR_submission', 'toy', 'xmonet', 'K{}', 'T4.json')
for i in [4]:
    with open(train_experiment_template.format(i)) as json_file:
        train_experiment_dict = json.load(json_file)

    for i in range(1):
        routines_extended.run_gridsearch(train_experiment_dict,
                                         hyperparams,
                                         constraints=constraints,
                                         retrain=True)

"""
test_experiment_file = os.path.join('..', 'test', 'K4_test.json')
routines_extended.evaluate_test_score_from_gridsearch(test_experiment_file,
                                                      train_experiment_file,
                                                      'th%mean_accuracy')
"""
