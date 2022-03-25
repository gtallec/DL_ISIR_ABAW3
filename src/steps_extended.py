import tensorflow as tf
import sys
import pandas as pd

def model_train_step(model, data_processing, optimizers):
    @tf.function
    def first_train_step(inputs, targets, loss_function, metrics):
        inputs_augmented, targets_augmented = data_processing(inputs, y=targets, training=True)
        y_pred = model(inputs_augmented, y=targets, training=True)
        y_pred['inputs_augmented'] = inputs_augmented
        y_pred['targets_augmented'] = targets_augmented
        metrics.update_state(targets, y_pred)

    @tf.function
    def train_step(inputs, targets, loss_function, metrics, grad_trackers):
        inputs_augmented, targets_augmented = data_processing(inputs, y=targets, training=True)
        with tf.GradientTape() as tape:
            y_pred = model(inputs_augmented, y=targets, training=True)
            loss_value = loss_function(targets_augmented, y_pred)
        opt_dict, trainable_variables = model.get_trainable_variables()
        tracked_variables = y_pred.get('track_grad', {})
        variables = dict({'trainable': trainable_variables,
                          'tracked': tracked_variables})
        grads = tape.gradient(loss_value, variables)
        for block in trainable_variables:
            block_optimizer = optimizers[opt_dict[block]]
            block_grads = grads['trainable'][block]
            block_variables = trainable_variables[block]
            
            """
            tf.print("########### VARIABLE of {} ############".format(block))
            for i in range(len(block_variables)):
                variable = block_variables[i]
                grad = block_grads[i]
                tf.print(variable.name, variable.shape, tf.math.reduce_mean(grad))
            """
            block_optimizer.apply_gradients(zip(block_grads, block_variables))

        grad_trackers.update_state(targets, grads['tracked'])
        y_pred['inputs_augmented'] = inputs_augmented
        y_pred['targets_augmented'] = targets_augmented
        metrics.update_state(targets_augmented, y_pred)


    def general_train_step(inputs, targets, loss_function, metrics, grad_trackers, first_step):
        if first_step:
            first_train_step(inputs=inputs,
                             targets=targets,
                             loss_function=loss_function,
                             metrics=metrics)
        else:
            train_step(inputs=inputs,
                       targets=targets,
                       loss_function=loss_function,
                       metrics=metrics,
                       grad_trackers=grad_trackers)

    return general_train_step

def model_test_step(model, data_processing):
    @tf.function
    def test_step(inputs, targets, metrics):
        inputs_augmented, targets_augmented = data_processing(inputs, y=targets, training=False)
        y_pred = model(inputs_augmented, training=False)
        y_pred['inputs_augmented'] = inputs_augmented
        y_pred['targets_augmented'] = targets_augmented
        metrics.update_state(targets, y_pred)
    return test_step

def model_eval_step(model, data_processing):
    @tf.function
    def eval_step(inputs, targets, metrics):
        inputs_augmented, targets_augmented = data_processing(inputs, y=targets, training=False)
        y_pred = model(inputs_augmented, training=False)
        y_pred['inputs_augmented'] = inputs_augmented
        y_pred['targets_augmented'] = targets_augmented
        metrics.update_state(targets, y_pred)
    return eval_step
