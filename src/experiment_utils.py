import json
import os
import copy
import subprocess

def duplicate_experiment(experiment_path, n_repeat):
    with open(experiment_path, 'r') as json_file:
        json_settings = json.load(json_file)

    experiment_dir = os.path.dirname(experiment_path)
    experiment_file = os.path.basename(experiment_path)
    experiment_name = os.path.splitext(experiment_file)
    experiment_base = experiment_name[0]
    experiment_extension = experiment_name[1]

    log_folder = json_settings.get('log_folder')
    experiments_dir = os.path.join(experiment_dir, experiment_base)

    if not os.path.isdir(experiments_dir):
        os.mkdir(experiments_dir)

    for i in range(1, n_repeat + 1):
        experiment_i_dict = copy.deepcopy(json_settings)
        experiment_i_dict['log_folder'] = os.path.join(log_folder, experiment_base + '_{}'.format(i))
        experiment_i_file = os.path.join(experiments_dir, experiment_base + '_{}'.format(i) + experiment_extension)
        with open(experiment_i_file, 'w') as json_file:
            json.dump(experiment_i_dict, json_file)
def get_experiment_folder(experiment_path):
    experiment_dir = os.path.dirname(experiment_path)
    experiment_file = os.path.basename(experiment_path)
    experiment_name = os.path.splitext(experiment_file)
    experiment_base = experiment_name[0]

    return os.path.join(experiment_dir, experiment_base)

def launch_experiment_folderv2(experiment_folder, launch_script, launch_args):
    experiments = os.listdir(experiment_folder)
    for experiment in experiments:
        experiment_path = os.path.join(experiment_folder,
                                       experiment)
        launch_cmd = (['python', launch_script]
                      +
                      ['-e', experiment_path]
                      +
                      launch_args)
        subprocess.Popen(args=launch_cmd,
                         shell=False).wait()

def unroll_cross_validation(experiment, experiment_file):
    meta_experiment = experiment['experiment']
    folds = meta_experiment['folds']
    train_fold_combinations = []
    eval_fold_combinations = []
    for i in range(folds):
        train_fold_combination = ([j for j in range(i)]
                                  +
                                  [j for j in range(i + 1, folds)])
        eval_fold_combination = [i]

        train_fold_combinations.append(train_fold_combination)
        eval_fold_combinations.append(eval_fold_combination)


    experiment_folder = os.path.splitext(experiment_file)[0]
    if not(os.path.exists(experiment_folder)):
        os.mkdir(experiment_folder)

    dataset_meta = experiment['dataset']
    log_folder = experiment.pop('log_folder')

    for i in range(folds):
        fold_i_experiment = copy.deepcopy(experiment)

        train_dataset_meta = copy.deepcopy(dataset_meta)
        train_dataset_meta['fold'] = train_fold_combinations[i]
        train_dataset_meta['mode'] = 'train'

        eval_dataset_meta = copy.deepcopy(dataset_meta)
        eval_dataset_meta['fold'] = eval_fold_combinations[i]
        eval_dataset_meta['mode'] = 'eval'
        fold_i_experiment['train_dataset'] = train_dataset_meta
        fold_i_experiment['eval_dataset'] = eval_dataset_meta
        fold_i_experiment['log_folder'] = os.path.join(log_folder, 'fold_{}'.format(i))
        fold_i_experiment_file = os.path.join(experiment_folder, 'fold_{}.json'.format(i))
        with open(fold_i_experiment_file, 'w') as json_file:
            json.dump(fold_i_experiment, json_file)

    return experiment_folder

def unroll_double_cross_validation(experiment_file):

    with open(experiment_file) as json_file:
        experiment = json.load(json_file)

    experiment_folder = os.path.splitext(experiment_file)[0]
    if not(os.path.isdir(experiment_folder)):
        os.mkdir(experiment_folder)

    meta_experiment = experiment['experiment']
    folds = meta_experiment['folds']
    subfolds = meta_experiment['subfolds']

    train_folds, eval_folds = fold_combination(folds)
    train_subfolds, eval_subfolds = fold_combination(subfolds)


    dataset_meta = experiment['dataset']
    log_folder = experiment.pop('log_folder')

    for i in range(folds):
        experiment_fold_i = copy.deepcopy(experiment)

        train_dataset_meta_fold_i = copy.deepcopy(dataset_meta)
        train_dataset_meta_fold_i['fold'] = train_folds[i]

        experiment_fold_i['train_dataset'] = train_dataset_meta_fold_i
        
        eval_dataset_meta_fold_i = copy.deepcopy(dataset_meta)
        eval_dataset_meta_fold_i['fold'] = eval_folds[i]
        experiment_fold_i['eval_dataset'] = eval_dataset_meta_fold_i

        log_folder_fold_i = os.path.join(log_folder,
                                         'fold_{}'.format(i))

        experiment_fold_i['log_folder'] = os.path.join(log_folder_fold_i)

        experiment_file_fold_i = os.path.join(experiment_folder, 'fold_{}'.format(i) + '.' + 'json')
        experiment_folder_fold_i = os.path.join(experiment_folder,
                                                'fold_{}'.format(i))
        if not(os.path.isdir(experiment_folder_fold_i)):
            os.makedirs(experiment_folder_fold_i)


        with open(experiment_file_fold_i, 'w') as json_file:
            json.dump(experiment_fold_i,
                      json_file,
                      indent=1)

        for j in range(subfolds):

            experiment_fold_ij = copy.deepcopy(experiment)

            train_dataset_meta_fold_ij = copy.deepcopy(train_dataset_meta_fold_i)
            train_dataset_meta_fold_ij['subfold'] = train_subfolds[j]
            experiment_fold_ij['train_dataset'] = train_dataset_meta_fold_ij

            eval_dataset_meta_fold_ij = copy.deepcopy(train_dataset_meta_fold_i)
            eval_dataset_meta_fold_ij['subfold'] = eval_subfolds[j]
            experiment_fold_ij['eval_dataset'] = eval_dataset_meta_fold_ij

            log_folder_fold_ij = os.path.join(log_folder_fold_i,
                                              'subfold{}'.format(j))
            experiment_file_fold_ij = os.path.join(experiment_folder_fold_i, 'subfold{}'.format(j) + '.' + 'json')

            with open(experiment_file_fold_ij, 'w') as json_file:
                json.dump(experiment_fold_ij,
                          json_file,
                          indent=1)

if __name__ == '__main__':
    experiment = {'experiment': {'folds': 3,
                                 'subfolds': 3},
                  'dataset': {'type' : 'disfa'},
                  'log_folder': 'test'}
    experiment_file = 'test.json'

    with open(experiment_file, 'w') as json_file:
        json.dump(experiment, json_file, indent=1)

    unroll_double_cross_validation(experiment_file)







        

