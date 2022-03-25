import sys
import copy as cp
import numpy as np

from tuning.model_config import MODEL_REQUIRED_PARAMS
from tuning.metric_config import METRIC_REQUIRED_PARAMS
from tuning.loss_config import LOSS_REQUIRED_PARAMS
from tuning.dataset_config import DATASET_REQUIRED_PARAMS
from tuning.opt_config import CONSTANT_OPT_REQUIRED_PARAMS, SCHEDULE_REQUIRED_PARAMS, CONDITION_REQUIRED_PARAMS, SCHEDULABLE_OPT_REQUIRED_PARAMS, EXTERNAL_SCHEDULE_REQUIRED_PARAMS
from tuning.callback_config import CALLBACK_REQUIRED_PARAMS
from tuning.data_processing_config import DATA_PROCESSING_REQUIRED_PARAMS

def get_required_parameters(model_dict):
    required_parameters = []
    for block in model_dict:
        block_type = block['type']
        block_required_params = MODEL_REQUIRED_PARAMS[block_type]
        required_parameters = required_parameters + block_required_params.keys()
    return required_parameters
    
def fill_model_meta(model_dict, hyperparams):
    filled_model_dict = cp.deepcopy(model_dict) 
    for block in filled_model_dict: 
        block_type = filled_model_dict[block]['type']
        required_params = MODEL_REQUIRED_PARAMS.get(block_type, {})
        for required_param in required_params:
            current_depth = filled_model_dict[block]
            required_param_value = hyperparams.get(required_param)
            if required_param not in hyperparams:
                sys.exit('{} should be specified in {}'.format(required_param, block))
            required_param_list = required_params[required_param]
            if not(isinstance(required_param_list, list)):
                current_depth[required_param_list] = required_param_value
            else:
                for i in range(len(required_param_list) - 1):
                    depth_key = required_param_list[i] 
                    if not(depth_key in current_depth):
                        current_depth[depth_key] = dict()
                    current_depth = current_depth[depth_key]
                current_depth[required_param_list[-1]] = required_param_value
    return filled_model_dict

def fill_metrics(metrics_list, hyperparams):
    filled_metrics_list = cp.deepcopy(metrics_list)
    for metric in filled_metrics_list:
        metric_type = metric['type']
        # List of required_params
        required_params = METRIC_REQUIRED_PARAMS.get(metric_type, [])
        for required_param in required_params:
            if required_param not in metric:
                required_param_value = hyperparams.get(required_param, None)
                if required_param_value is None:
                    sys.exit('{} should be specified in {}'.format(required_param, metric_type))
                metric[required_param] = required_param_value
    return filled_metrics_list

def fill_dataset(dataset_dict, hyperparams):
    filled_dataset_dict = cp.deepcopy(dataset_dict)
    dataset_type = filled_dataset_dict.get('type', filled_dataset_dict.get('name'))
    required_params = DATASET_REQUIRED_PARAMS.get(dataset_type, [])
    for required_param in required_params:
        if required_param not in dataset_dict:
            required_param_value = hyperparams.get(required_param, None)
            if required_param_value is None:
                sys.exit('{} should be specified in {}'.format(required_param, dataset_type))
            filled_dataset_dict[required_param] = required_param_value
    return filled_dataset_dict

def fill_losses(loss_list, hyperparams):
    filled_loss_list = cp.deepcopy(loss_list)
    for loss in filled_loss_list:
        loss_type = loss['type']
        # List of required_params
        required_params = LOSS_REQUIRED_PARAMS.get(loss_type, [])
        if 'weight' not in loss:
            weight = hyperparams.get('weight_{}'.format(loss_type), None)
            if weight is None:
                sys.exit('{} should be specified in {}'.format('weight', loss_type))
            loss['weight'] = weight
        for required_param in required_params:
            if required_param not in loss:
                required_param_value = hyperparams.get(required_param, None)
                if required_param_value is None:
                    sys.exit('{} should be specified in {}'.format(required_param, loss_type))
                loss[required_param] = required_param_value
    return filled_loss_list

def fill_callbacks(callback_list, hyperparams):
    filled_callback_list = cp.deepcopy(callback_list)
    for callback in filled_callback_list:
        callback_type = callback['type']
        required_params = CALLBACK_REQUIRED_PARAMS.get(callback_type, {})
        for required_param in required_params:
            if required_param not in callback:
                required_param_value = hyperparams.get(required_param, None)
                if required_param_value is None:
                    sys.exit('{} should be specified in {}'.format(required_param, callback_type))
                callback[required_params[required_param]] = required_param_value
    return filled_callback_list

def fill_batchsize_meta(experiment_dict, hyperparams):
    filled_experiment_dict = cp.deepcopy(experiment_dict)
    modes = ['train', 'test', 'eval']
    for mode in modes:
        mode_dataset = '{}_dataset'.format(mode)
        if mode_dataset in experiment_dict:
            filled_experiment_dict[mode_dataset]['batchsize'] = hyperparams['batchsize']
    return filled_experiment_dict

def fill_opt_meta(opt_dict, hyperparams):
    filled_opt_dict = cp.deepcopy(opt_dict)
    for opt_name in filled_opt_dict:
        opt_params = filled_opt_dict[opt_name]['params']
        opt_type = opt_params['type']

        required_schedulable_opt_params = SCHEDULABLE_OPT_REQUIRED_PARAMS[opt_type]
        required_constant_opt_params = CONSTANT_OPT_REQUIRED_PARAMS[opt_type]

        for key in required_schedulable_opt_params:
            required_hyperparam_key = required_schedulable_opt_params[key]
            hyperparam_key = '{}0_{}'.format(required_hyperparam_key, opt_name)
            if not(hyperparam_key in hyperparams):
                sys.exit('Missing {}'.format(hyperparam_key))
            opt_params[key] = hyperparams[hyperparam_key]

        for key in required_constant_opt_params:
            hyperparam_key = required_constant_opt_params[key] + '_' + opt_name
            if not(hyperparam_key in hyperparams):
                sys.exit('Missing {}'.format(hyperparam_key))
            opt_params[key] = hyperparams[hyperparam_key]

        if 'schedules' in filled_opt_dict[opt_name]:
            schedules = filled_opt_dict[opt_name]['schedules']
            for i in range(len(schedules)):
                schedule = schedules[i]
                schedule_type = schedule['type'].split('_')[-1]
                schedule_args = schedule['schedule_args']
                for j in range(len(schedule_args)):
                    schedule_arg = schedule_args[j]
                    schedule_arg_type = schedule_arg['type']
                    required_schedule_params = SCHEDULE_REQUIRED_PARAMS.get(schedule_arg_type, [])
                    required_external_schedule_params = EXTERNAL_SCHEDULE_REQUIRED_PARAMS.get(schedule_arg_type, [])
                    schedule_arg['value'] = hyperparams.get(schedule_type + str(j) + '_' + opt_name)
                    for required_schedule_param in required_schedule_params:
                        key = schedule_type + str(j) + '_' + required_schedule_param + '_' + opt_name
                        schedule_arg[required_schedule_param] = hyperparams[key]
                    for required_external_schedule_param in required_external_schedule_params:
                        schedule_arg[required_external_schedule_param] = hyperparams[required_external_schedule_param]
 
                if len(schedule_args) > 1:
                    condition_args = schedule['condition_args']
                    for j in range(len(condition_args)):
                        condition_arg = condition_args[j]
                        condition_arg_type = condition_arg['type']
                        required_condition_params = CONDITION_REQUIRED_PARAMS[condition_arg_type]
                        for required_condition_param in required_condition_params:
                            key = schedule_type + str(j) + '_' + required_condition_param + '_' + opt_name
                            condition_arg[required_condition_param] = hyperparams[key]

    return filled_opt_dict

def fill_data_processing(data_processing_list, hyperparams):
    filled_data_processing_list = cp.deepcopy(data_processing_list)
    for data_processing in filled_data_processing_list:
        data_processing_type = data_processing['type']
        required_params = DATA_PROCESSING_REQUIRED_PARAMS.get(data_processing_type, [])
        for required_param in required_params:
            if required_param not in data_processing:
                required_param_value = hyperparams.get(required_param, None)
                if required_param_value is None:
                    sys.exit('{} should be specified in {}'.format(required_param, data_processing_type))
                data_processing[required_param] = required_param_value
    return filled_data_processing_list


def fill_meta(experiment_dict, hyperparams):
    filled_experiment_dict = cp.deepcopy(experiment_dict)
    model_experiment_dict = filled_experiment_dict['model']
    optimizers_experiment_dict = filled_experiment_dict['optimizers']
    losses_experiment_dict = filled_experiment_dict['losses']
    callbacks_experiment_dict = filled_experiment_dict['callbacks']
    data_processing_list = filled_experiment_dict['data_processing']

    filled_experiment_dict['model'] = fill_model_meta(model_experiment_dict, hyperparams)
    filled_experiment_dict['optimizers'] = fill_opt_meta(optimizers_experiment_dict, hyperparams)
    filled_experiment_dict['losses'] = fill_losses(losses_experiment_dict, hyperparams)
    filled_experiment_dict['callbacks'] = fill_callbacks(callbacks_experiment_dict, hyperparams)
    filled_experiment_dict['data_processing'] = fill_data_processing(data_processing_list, hyperparams) 

    filled_experiment_dict = fill_batchsize_meta(filled_experiment_dict, hyperparams)

    for mode in ['train', 'test', 'eval']:
        metric = '{}_metrics'.format(mode)
        if metric in filled_experiment_dict:
            filled_experiment_dict[metric] = fill_metrics(filled_experiment_dict.get(metric, {}), hyperparams)

        dataset = '{}_dataset'.format(mode)
        if dataset in filled_experiment_dict:
            filled_experiment_dict[dataset] = fill_dataset(filled_experiment_dict.get(dataset, {}), hyperparams)


    if 'epochs' in hyperparams:
        filled_experiment_dict['epochs'] = hyperparams['epochs']

    return filled_experiment_dict


def unroll_constraint(hyperparams, constraints):
    unrolled_hyperparams = cp.deepcopy(hyperparams)
    for constraint in constraints:
        constrained_dict = dict()
        for hyperparam in unrolled_hyperparams:
            vals = unrolled_hyperparams[hyperparam]
            if constraint in hyperparam:
                for constrained in constraints[constraint]:
                    new_hyperparam = hyperparam.replace(constraint, constrained)
                    constrained_dict[new_hyperparam] = vals
            else:
                constrained_dict[hyperparam] = vals
        unrolled_hyperparams = constrained_dict
    return unrolled_hyperparams


def group_hyperparams(hyperparams):
    grouped_hyperparams = dict()
    group_of_hyperparams = dict()
    for key in hyperparams:
        split_on_underscore = key.split('_')
        if split_on_underscore[-1].isdigit():
            group_name = '_'.join(split_on_underscore[:-1]) 
            if not(group_name in group_of_hyperparams):
                group_of_hyperparams[group_name] = 1
            else:
                group_of_hyperparams[group_name] += 1
        else:
            grouped_hyperparams[key] = hyperparams[key]
    for group_name in group_of_hyperparams:
        grouped_hyperparams[group_name] = [hyperparams[group_name + '_' + str(i)] for i in range(group_of_hyperparams[group_name])]
    return grouped_hyperparams
 
def get_hyperparam_grid(hyperparams):
    # Meta for flattening
    flat_list = []
    ordered_keys = list(hyperparams.keys())
    for key in ordered_keys:
        flat_list.append(np.arange(len(hyperparams[key])))
    
    grid_coords = np.concatenate([np.expand_dims(a, axis=-1) for a in np.meshgrid(*flat_list)], axis=-1)
    grid_coords = grid_coords.reshape(np.prod(grid_coords.shape[:-1]), grid_coords.shape[-1])
   
    hyperparams_dict_list = []
    for i in range(grid_coords.shape[0]):
        hyperparams_dict = dict()
        for j, key in enumerate(ordered_keys):
            hyperparams_dict[key] = hyperparams[key][grid_coords[i, j]]
        hyperparams_dict_list.append(hyperparams_dict)
    return hyperparams_dict_list    
