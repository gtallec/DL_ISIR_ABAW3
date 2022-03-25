import json
import copy
import sys
import os

import pandas as pd
import numpy as np

from models.interface import get_model
from dataset_statistics import get_dataset_statistic

import metrics_extended.interface
import losses_extended.interface

# TODO : gen_dataset should be renamed interface and put in the dataset folder
from gen_datasets import gen_dataset, df_columns_dataset
from models.data_processing.interface import DataProcessing


from configs.config import LOG_PATH, RESULT_PATH

from optimizers import optimizer
from callbacks.callback_builder import CallbackBuilder





class ExperimentParser:
    """ Abstract class for mapping from experiment dictionary to keras/tf usable objects """

    def __init__(self, experiment_dict, supported_modes, verbose=False):
        self.experiment_dict = experiment_dict
        self.supported_modes = supported_modes
        self.base_log_folder = os.path.join(LOG_PATH, self.experiment_dict.get('log_folder'))

        self.dataset_statistics = self.experiment_dict.get('dataset_statistics', dict())
        self.computed_dataset_statistics = dict()

        for mode in self.supported_modes:
            self.computed_dataset_statistics[mode] = dict()
            self.compute_dataset_statistics(mode=mode)

        self.verbose = verbose
        self.parse_experiment()

    def parse_experiment(self):
        pass

    def get_epochs(self):
        return self.experiment_dict.get('epochs')

    """
    def parse_model(self):
        model_dict = self.experiment_dict.get('model')
        model = get_model(model_dict,
                          **self.computed_dataset_statistics[self.supported_modes[0]],
                          training_dataset=self.get_dataset(mode='train', meta=False))
        dataset_element_size = self.experiment_dict.get('dataset', {}).get('size', (299, 299, 3))
        model.build((None, *dataset_element_size))
        model.load_pretrained_weights()
        self.model = model
        if self.verbose:
            self.model.summary()
    """

    def get_metrics(self, mode):
        if mode not in self.supported_modes:
            sys.exit('mode not in supported modes in get_metrics function : ' + mode)
        metrics_kwarg_list = self.experiment_dict.get('{}_metrics'.format(mode),
                                                      self.experiment_dict.get('metrics', dict({})))
        cp_metrics_kwarg_list = copy.deepcopy(metrics_kwarg_list)
        for metrics_kwarg in cp_metrics_kwarg_list:
            metrics_kwarg.update(self.computed_dataset_statistics[mode])

        dataset_columns = df_columns_dataset(self.get_dataset_meta())
        return metrics_extended.interface.get_metrics(cp_metrics_kwarg_list,
                                                      dataset_columns,
                                                      log_folder=self.get_log_folder())

    def get_dataset(self, mode, meta=False):
        if mode not in self.supported_modes:
            sys.exit('mode not in supported modes in get_dataset function : ' + mode)
        dataset_meta = self.experiment_dict['{}_dataset'.format(mode)]
        dataset_meta['meta'] = meta
        return gen_dataset(dataset_meta)

    def get_dataset_meta(self):
        return self.experiment_dict['dataset']

    def compute_dataset_statistics(self, mode):
        dataset_statistics = self.dataset_statistics.get(mode, [])
        if len(dataset_statistics) > 0:
            meta_dataset = self.get_dataset(mode, meta=True)
            for dataset_statistic in dataset_statistics:
                computed_statistic = get_dataset_statistic(dataset_statistic)(meta_dataset)
                self.computed_dataset_statistics[mode][dataset_statistic] = computed_statistic

    def get_dataset_statistics(self, mode):
        return self.computed_dataset_statistics[mode]

    def get_model(self):
        return self.model

    def get_data_processing(self):
        data_processing_list = self.experiment_dict.get('data_processing')
        return DataProcessing(data_processing_list, dataset=self.get_dataset(mode='train', meta=False))

    def get_pre_callbacks(self):
        return self.experiment_dict.get('callbacks', [])

    def get_log_folder(self):
        return os.path.join(LOG_PATH, self.experiment_dict.get('log_folder'))

    def set_log_suffix(self, suffix):
        self.experiment_dict['log_folder'] = os.path.join(self.base_log_folder, suffix)

    def reset_log_suffix(self):
        self.experiment_dict['log_folder'] = self.base_log_folder

    def get_tracking_path(self):
        return os.path.join(self.base_log_folder, 'tracking.csv')

    def get_storing_path(self):
        return os.path.join(RESULT_PATH, self.experiment_dict.get('storing'))

    def get_weight_path(self):
        return os.path.join(self.get_log_folder(), 'weights', 'ckpt')

    def get_checkpoints_path(self):
        # Get checkpoint frequency and template file
        epochs = self.get_epochs()
        callback_list = self.get_pre_callbacks()
        found_checkpoint = False
        i = 0
        while not(found_checkpoint) and i < len(callback_list):
            callback = callback_list[i]
            found_checkpoint = (callback.get('type') == 'ckpt')
            i += 1

        if i <= len(callback_list):
            save_rate = callback['save_rate']
            ckpt_template = callback['ckpt_template']
            epoch_checkpoints = [save_rate * i for i in range(1, (epochs // save_rate) + 1)]
            weight_paths = [os.path.join(self.get_log_folder(), 'checkpoints', ckpt_template.format(epoch_checkpoint)) for epoch_checkpoint in epoch_checkpoints]

        # epoch_checkpoints = epoch_checkpoints  + [epochs]
        # weight_paths = weight_paths + [self.get_weight_path()]
        return epoch_checkpoints, weight_paths



class TrainExperimentParser(ExperimentParser):
    """ Class that converts train experiment dictionary into keras/tf usable objects"""

    def __init__(self, experiment_dict, verbose=False):
        super(TrainExperimentParser, self).__init__(experiment_dict=experiment_dict,
                                                    supported_modes=['train', 'test', 'eval'],
                                                    verbose=verbose)

    def parse_experiment(self):
        self.parse_model()
        self.parse_optimizers()
        self.parse_losses()
        self.parse_callbacks()

    def parse_model(self):
        model_dict = self.experiment_dict.get('model')
        model = get_model(model_dict,
                          **self.computed_dataset_statistics[self.supported_modes[0]],
                          train_dataset=self.get_dataset(mode='train', meta=False))
        dataset_element_size = self.experiment_dict.get('dataset', {}).get('size', (299, 299, 3))
        model.build((None, *dataset_element_size))
        model.load_pretrained_weights()
        self.model = model
        if self.verbose:
            self.model.summary()

    def parse_optimizers(self):
        """Parsing optimizers all keras optimizers + AdamW are supported for now"""
        optimizer_args = self.experiment_dict.get('optimizers')
        optimizers_dict = dict()

        callback_list = self.experiment_dict["callbacks"]
        for optimizer_name in optimizer_args:
            optimizer_dict = optimizer_args.get(optimizer_name)
            optimizer_params = optimizer_dict['params']
            optimizer_instance = optimizer(optimizer_params)

            schedules = optimizer_dict['schedules']
            for schedule in schedules:
                schedule["optimizer"] = optimizer_instance
                schedule_args = schedule["schedule_args"]
                for i in range(len(schedule_args)):
                    schedule_arg = schedule_args[i]
                    schedule_arg = dict(**schedule_arg,
                                        **self.get_dataset_statistics('train'))
                    schedule_args[i] = schedule_arg
                callback_list.append(schedule)

            optimizers_dict[optimizer_name] = optimizer_instance
        self.experiment_dict['callbacks'] = callback_list
        self.optimizers = optimizers_dict

    def parse_callbacks(self):
        callback_builder = CallbackBuilder(self)
        self.callback_list = callback_builder.build_callback_list()

    def parse_losses(self):
        loss_kwarg_list = self.experiment_dict.get('losses')
        cp_loss_kwarg_list = copy.deepcopy(loss_kwarg_list)
        for cp_loss_kwarg in cp_loss_kwarg_list:
            cp_loss_kwarg.update(self.computed_dataset_statistics['train'])
        loss = losses_extended.interface.get_losses(cp_loss_kwarg_list)
        self.losses = loss

    def get_callbacks(self):
        return self.callback_list

    def get_losses(self):
        return self.losses 

    def get_optimizers(self):
        return self.optimizers

    def get_gradient_trackers(self):
        metrics_kwarg_list = self.experiment_dict.get('gradient_trackers', dict({}))
        dataset_columns = df_columns_dataset(self.get_dataset_meta())
        return metrics_extended.interface.get_metrics(copy.deepcopy(metrics_kwarg_list),
                                                      dataset_columns,
                                                      log_folder=self.get_log_folder())

class EvalExperimentParser(ExperimentParser):
    """ Class that converts eval experiment dictionary into keras/tf usable objects"""

    def __init__(self, experiment_dict):
        super(EvalExperimentParser, self).__init__(experiment_dict=experiment_dict,
                                                   supported_modes=['eval'])

    def parse_model(self):
        model_dict = self.experiment_dict.get('model')
        model = get_model(model_dict,
                          **self.computed_dataset_statistics[self.supported_modes[0]])
        dataset_element_size = self.experiment_dict.get('dataset', {}).get('size', (299, 299, 3))
        model.build((None, *dataset_element_size))
        model.load_pretrained_weights()
        self.model = model
        if self.verbose:
            self.model.summary()

    def get_threshold_metric(self):
        metric_kwargs_list = self.experiment_dict.get('eval_metrics')
        found = False
        i = 0
        while (i < len(metric_kwargs_list)) and not(found):
            metric_kwargs = metric_kwargs_list[i]
            found = (metric_kwargs['type'] == 'th_test')
            i += 1

        if found:
            return metric_kwargs_list[i-1]
        else:
            return None

    def fetch_best_train_threshold(self, epoch, metric_names):
        tracking_df = pd.read_csv(self.get_tracking_path())
        epoch_row = tracking_df.loc[tracking_df['epoch'] == epoch].iloc[0]
        threshold_by_metrics = []
        for metric_name in metric_names:
            threshold_by_metrics.append(np.expand_dims(np.load(epoch_row['test_th%mean_' + metric_name]), axis=0))
        return np.concatenate(threshold_by_metrics, axis=0)

    def fill_best_threshold(self, epoch=None):
        if epoch is None:
            epoch = self.get_epochs()
        threshold_metric = self.get_threshold_metric()
        if not(threshold_metric is None):
            metric_names = threshold_metric['metric_names']
            best_thresholds = self.fetch_best_train_threshold(epoch=epoch,
                                                              metric_names=metric_names)
            threshold_metric['thresholds'] = best_thresholds

    def set_best_threshold(self, thresholds):
        threshold_metric = self.get_threshold_metric()
        if not(threshold_metric is None):
            threshold_metric['thresholds'] = thresholds


class MetaExperimentParser(ExperimentParser):
    """ Class that converts meta experiment_dictionary into keras/tf usable objects"""

    def __init__(self, experiment_dict):
        super(MetaExperimentParser, self).__init__(experiment_dict=experiment_dict,
                                                   supported_modes=[])

    def get_training_routine(self):
        return self.experiment_dict.get('training_routine', dict({}))

    def get_storing_path(self):
        return os.path.join(RESULT_PATH, self.experiment_dict.get('storing'))


class VisualExperimentParser(ExperimentParser):
    """ Class that converts eval experiment_dictionary into keras/tf usable objects"""

    def __init__(self, experiment_dict):
        super(VisualExperimentParser, self).__init__(experiment_dict=experiment_dict,
                                                     supported_modes=[])
