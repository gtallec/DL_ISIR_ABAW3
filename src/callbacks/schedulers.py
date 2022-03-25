from callbacks.callbacks import Callback
from callbacks.schedules import schedule
from callbacks.conditions import condition

from tensorflow.python.keras import backend as K

import tensorflow as tf

import copy



class OptimizerHyperScheduler(Callback):
    def __init__(self, optimizer, hyper, schedule_args, reset, condition_args=[], verbose=False, step_on='epoch', **kwargs):
        super(OptimizerHyperScheduler, self).__init__(**kwargs)
        self.optimizer = optimizer

        self.schedules = [schedule(schedule_arg) for schedule_arg in schedule_args]
        self.conditions = [condition(condition_arg) for condition_arg in condition_args]
        self.hyper = hyper
        self.verbose = verbose
       
        self.current_relative_step = 0
        self.current_step = 0
        self.current_condition_id = 0
        self.reset = reset

        self.step_on = step_on

    def update_step(self, logs=None):
        # epoch = logs['epoch'].iloc[0]
        if ((self.current_condition_id < len(self.conditions)) and self.conditions[self.current_condition_id].check(logs=logs)):
            self.current_condition_id += 1
            if self.verbose:
                print('current_condition_id : ', self.current_condition_id)
            self.current_relative_step = self.current_step
            if self.reset:
                for var in self.optimizer.variables():
                    var.assign(tf.zeros_like(var))

        old_hyper_value = float(K.get_value(self.optimizer._get_hyper(self.hyper)))
        new_hyper_value = self.schedules[self.current_condition_id](step=self.current_step - self.current_relative_step,
                                                                    val=old_hyper_value)
        self.optimizer._set_hyper(self.hyper, new_hyper_value)
        self.current_step += 1

    def on_epoch_begin(self, logs=None):
        if self.step_on == 'epoch':
            self.update_step(logs=logs)
        else:
            pass

    def on_batch_begin(self, batch, logs=None):
        if self.step_on == 'batch':
            self.update_step(logs=logs)
        else:
            pass


class AdamWScheduler(Callback):
    def __init__(self, optimizer, schedule_args, condition_args=[], step_on='epoch', reset=False, verbose=False, **kwargs):
        super(AdamWScheduler, self).__init__(**kwargs)
        self.learning_rate_scheduler = LearningRateScheduler(optimizer=optimizer,
                                                             schedule_args=copy.deepcopy(schedule_args),
                                                             condition_args=copy.deepcopy(condition_args),
                                                             step_on=step_on,
                                                             reset=reset,
                                                             verbose=verbose)
        self.weight_decay_scheduler = WeightDecayScheduler(optimizer=optimizer,
                                                           schedule_args=copy.deepcopy(schedule_args),
                                                           condition_args=copy.deepcopy(condition_args),
                                                           step_on=step_on,
                                                           reset=reset,
                                                           verbose=verbose)

    def on_epoch_begin(self, epoch, logs=None):
        self.learning_rate_scheduler.on_epoch_end(epoch, logs=logs)
        self.weight_decay_scheduler.on_epoch_end(epoch, logs=logs)

class LearningRateScheduler(OptimizerHyperScheduler):
    def __init__(self, optimizer, schedule_args, condition_args=[], step_on='epoch', reset=False, verbose=False, **kwargs):
        super(LearningRateScheduler, self).__init__(optimizer=optimizer,
                                                    hyper='learning_rate',
                                                    step_on=step_on,
                                                    schedule_args=schedule_args,
                                                    condition_args=condition_args,
                                                    reset=reset,
                                                    verbose=verbose,
                                                    **kwargs)
class WeightDecayScheduler(OptimizerHyperScheduler):
    def __init__(self, optimizer, schedule_args, condition_args=[], step_on='epoch', reset=False, verbose=False, **kwargs):
        super(WeightDecayScheduler, self).__init__(optimizer=optimizer,
                                                   hyper='weight_decay',
                                                   step_on=step_on,
                                                   schedule_args=schedule_args,
                                                   condition_args=condition_args,
                                                   reset=reset,
                                                   verbose=verbose,
                                                   **kwargs)


SUPPORTED_SCHEDULERS = {"schedule_lr": LearningRateScheduler,
                        "schedule_wd": WeightDecayScheduler,
                        "schedule_adamw": AdamWScheduler}
