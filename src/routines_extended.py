import os
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from steps_extended import model_train_step, model_test_step, model_eval_step
from experiment_parser_extended import TrainExperimentParser, EvalExperimentParser

def train(experiment_dict, verbose=False):
    experiment = TrainExperimentParser(experiment_dict, verbose=verbose)

    #########################
    # Datasets and metrics
    # TRAIN
    train_dataset = experiment.get_dataset('train')
    train_metrics = experiment.get_metrics('train')
    grad_trackers = experiment.get_gradient_trackers()

    # TEST
    has_test_step = ('test_dataset' in experiment_dict)
    if has_test_step:
        test_dataset = experiment.get_dataset('test')
        test_metrics = experiment.get_metrics('test')
        threshold_based_test, threshold_test_metric = test_metrics.get_metric('th_train')
    

    # EVAL
    eval_dataset = experiment.get_dataset('eval')
    eval_metrics = experiment.get_metrics('eval')
    threshold_based_eval, threshold_eval_metric = eval_metrics.get_metric('th_test') 

    model = experiment.get_model()
    
    data_processing = experiment.get_data_processing()
    optimizers = experiment.get_optimizers()
    loss = experiment.get_losses()
    epochs = experiment_dict['epochs']
    #########################
    # Callback_list
    callback_list = experiment.get_callbacks()

    #########################
    # Processing steps
    # TRAIN
    train_step = model_train_step(model, data_processing, optimizers)

    # TEST
    test_step = model_test_step(model, data_processing)
    #########################
    # Verbose Settings
    log_dfs = []
    if verbose:
        epoch_range = range(epochs + 1)
    else:
        epoch_range = tqdm(range(epochs + 1))


    callback_list.on_train_begin()
    for epoch in epoch_range:
        first_epoch = (epoch == 0)
        ##################################################
        # Training Pass
        if not(first_epoch):
            callback_list.on_epoch_begin()

        for (train_inputs, train_labels) in train_dataset:

            if not(first_epoch):
                callback_list.on_batch_begin(batch=(train_inputs, train_labels))

            train_step(inputs=train_inputs,
                       targets=train_labels,
                       loss_function=loss,
                       metrics=train_metrics,
                       grad_trackers=grad_trackers,
                       first_step=first_epoch)


            if not(first_epoch):
                callback_list.on_batch_end(batch=(train_inputs, train_labels))

        train_df = train_metrics.result_to_df()
        grad_df = grad_trackers.result_to_df()
        
        train_metrics.reset_states()
        grad_trackers.reset_states()
        ##################################################
        # Testing Pass
        if has_test_step:
            for (test_inputs, test_labels) in test_dataset:
                test_step(inputs=test_inputs,
                          targets=test_labels,
                          metrics=test_metrics)
            test_df = test_metrics.result_to_df()

            best_test_thresholds = threshold_test_metric.get_best_thresholds()

            best_test_thresholds = threshold_test_metric.get_best_thresholds()

            if threshold_based_test and threshold_based_eval:
                # (M, N)
                best_test_thresholds = threshold_test_metric.get_best_thresholds()
                threshold_eval_metric.set_thresholds(best_test_thresholds)
 
            test_metrics.reset_states()
        ##################################################
        # Eval Pass
        for (eval_inputs, eval_labels) in eval_dataset:
            test_step(inputs=eval_inputs, 
                      targets=eval_labels,
                      metrics=eval_metrics)
        eval_df = eval_metrics.result_to_df()
        eval_metrics.reset_states()
        ##################################################
        # LOG
        epoch_log_dfs = [grad_df]
        # TRAIN
        rename_train = dict()
        for train_name in train_df.columns:
            rename_train[train_name] = 'train_' + train_name
        train_df = train_df.rename(columns=rename_train)
        epoch_log_dfs.append(train_df)

        # TEST
        if has_test_step:
            rename_test = dict()
            for test_name in test_df.columns:
                rename_test[test_name] = 'test_' + test_name
            test_df = test_df.rename(columns=rename_test)
            epoch_log_dfs.append(test_df)

        # EVAL 
        rename_eval = dict()
        for eval_name in eval_df.columns:
            rename_eval[eval_name] = 'eval_' + eval_name
        eval_df = eval_df.rename(columns=rename_eval)
        epoch_log_dfs.append(eval_df)


        log_df = pd.concat(epoch_log_dfs, axis=1)

        if verbose:
            print(50 * '-')
            print('EPOCH {}'.format(epoch))
            for name in log_df.columns:
                print(name, ':', log_df.iloc[0][name])
            print(50 * '-')

        log_df['epoch'] = [epoch]
        log_dfs.append(log_df)


        if not(first_epoch):
            callback_list.on_epoch_end(logs=log_df)

    callback_list.on_train_end()

    # SAVE PATHS:
    tracking_path = experiment.get_tracking_path()
    weight_path = experiment.get_weight_path()

    if not(os.path.exists(weight_path)):
        os.makedirs(weight_path)

    model.save_weights(weight_path, save_format='tf')
    pd.concat(log_dfs, axis=0).to_csv(tracking_path, index=False)   
    return tracking_path

def evaluate_model_on_dataset(model, data_processing, dataset, metrics):
    eval_step = model_eval_step(model, data_processing)
    for (eval_inputs, eval_labels) in tqdm(dataset): 
        eval_step(inputs=eval_inputs,
                  targets=eval_labels,
                  metrics=metrics) 
    return metrics.result_to_df()

def evaluate(experiment_dict, model, fill_threshold=True):
    experiment = EvalExperimentParser(experiment_dict)

    if fill_threshold:
        experiment.fill_best_threshold()

    eval_dataset = experiment.get_dataset('eval')
    eval_metrics = experiment.get_metrics('eval')
    data_processing = experiment.get_data_processing()

    return evaluate_model_on_dataset(model=model,
                                     data_processing=data_processing,
                                     dataset=eval_dataset,
                                     metrics=eval_metrics)

