import tensorflow as tf
import pandas as pd
import os
import numpy as np

from metrics_extended.classification_functions import get_classification_function

class MaskedClassificationScores(tf.keras.metrics.Metric):
    def __init__(self, metric_names, thresholds, pred_in, dataset_columns, **kwargs): 
        """
        metric_names : A list of M elements with the name of the M metrics to track
        thresholds (M, N, T) : For each of the M metrics and each of the N coordinates, metrics are computed for
        T different thresholds
        """

        super(MaskedClassificationScores, self).__init__(**kwargs)
        self.metric_names = metric_names
        self.dataset_columns = dataset_columns
        self.pred_in = pred_in

        # (M, N, T)
        self.thresholds = tf.Variable(initial_value=thresholds,
                                      trainable=False)
        self.M = tf.shape(self.thresholds)[0]
        self.N = tf.shape(self.thresholds)[1]
        self.T = tf.shape(self.thresholds)[2]

        th_shape = tf.shape(self.thresholds)
        
        self.tp = self.add_weight(name='tp',
                                  initializer='zeros',
                                  shape=th_shape)

        self.tn = self.add_weight(name='tn',
                                  initializer='zeros',
                                  shape=th_shape)

        self.fp = self.add_weight(name='fp',
                                  initializer='zeros',
                                  shape=th_shape)

        self.fn = self.add_weight(name='fp',
                                  initializer='zeros',
                                  shape=th_shape)

        self.classification_functions = []
        for metric_names in self.metric_names:
            self.classification_functions.append(get_classification_function(metric_names))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.round(y_true)
        y_dict = y_pred[self.pred_in]
        # (B, S, N)
        pred = y_dict['prediction']
        # (B, S)
        mask = y_dict['padding_mask']

        # (B x S, N)
        y_true = tf.reshape(y_true, (-1, self.N))
        # (B x S, N)
        pred = tf.reshape(pred, (-1, self.N))

        BS = tf.shape(y_true)[0]

        # (B x S, M, N, T)
        thresholds = tf.tile(tf.expand_dims(self.thresholds, axis=0),
                             multiples=[BS, 1, 1, 1])
        # (B x S, M, N, T)
        pred = tf.tile(tf.reshape(pred, (BS, 1, self.N, 1)),
                       multiples=[1, self.M, 1, self.T])
        # (B x S, M, N, T)
        pred_bin = tf.dtypes.cast((pred - thresholds) > 0, dtype=tf.float32)

        # (B x S, M, N, T)
        y_true_bin = tf.tile(tf.reshape(y_true, (BS, 1, self.N, 1)),
                             multiples=[1, self.M, 1, self.T])

        # (B x S, 1, 1, 1) -> (B x S, M, N, T)
        mask = tf.reshape(mask, (-1, ))[:, tf.newaxis, tf.newaxis, tf.newaxis]

        # (B x S, M, N, T)
        tp = pred_bin * y_true_bin * mask
        tn = (1 - pred_bin) * (1 - y_true_bin) * mask
        fp = pred_bin * (1 - y_true_bin) * mask
        fn = (1 - pred_bin) * y_true_bin * mask

        # (M, N, T)
        self.tp.assign_add(tf.math.reduce_sum(tp, axis=0))
        self.tn.assign_add(tf.math.reduce_sum(tn, axis=0))
        self.fp.assign_add(tf.math.reduce_sum(fp, axis=0))
        self.fn.assign_add(tf.math.reduce_sum(fn, axis=0))

    def get_metric_names(self):
        return self.metric_names

    def reset_states(self):
        self.tp.assign(tf.zeros((self.M, self.N, self.T)))
        self.fp.assign(tf.zeros((self.M, self.N, self.T)))
        self.fn.assign(tf.zeros((self.M, self.N, self.T)))
        self.tn.assign(tf.zeros((self.M, self.N, self.T)))


class MaskedTrainClassificationScores(MaskedClassificationScores):
    def __init__(self, metric_names, threshold_step, n_coords, pred_in, dataset_columns, log_folder, **kwargs):
        super(MaskedTrainClassificationScores, self).__init__(metric_names=metric_names,
                                                              thresholds=tf.tile(tf.expand_dims(tf.expand_dims(threshold_step * tf.range(0, (1/threshold_step) + 1),
                                                                                                               axis=0), axis=0),
                                                                                 (1, n_coords, 1)),
                                                              pred_in=pred_in,
                                                              dataset_columns=dataset_columns)
        self.log_folder = log_folder
        if not(os.path.exists(self.log_folder)):
            os.makedirs(self.log_folder)
        self.threshold_step = threshold_step
        self.n_res = 0
        
        self.best_thresholds = self.add_weight(name='best_th',
                                               initializer='zeros',
                                               shape=(len(metric_names), n_coords))

    def result(self):
        result_list = []
        for i in range(len(self.metric_names)):
            result_list.append(tf.expand_dims(self.classification_functions[i](self.tp[0],
                                                                               self.tn[0],
                                                                               self.fp[0],
                                                                               self.fn[0]),
                                              axis=0))
        # (M, N, T) 
        results = tf.concat(result_list, axis=0)
        return results

    def result_to_df(self):
        # (T, )
        thresholds = self.threshold_step * tf.range(0, (1/self.threshold_step) + 1)
        # (M, T, N)
        results = tf.transpose(self.result(), perm=(0, 2, 1))
        # Compute the best threshold for each metric and coordinate
        # (M, N)
        best_result = tf.math.reduce_max(results, axis=1)
        mean_best_result = tf.math.reduce_mean(best_result, axis=1)
        # (M, N)
        best_thresholds_index = tf.math.argmax(results, axis=1)
        # (M, N)
        best_thresholds = tf.gather(thresholds, best_thresholds_index, axis=0)
        self.best_thresholds = best_thresholds

        results = results.numpy()
        best_result = best_result.numpy()
        mean_best_result = mean_best_result.numpy()
        thresholds = thresholds.numpy()
        best_thresholds = best_thresholds.numpy()

        metric_dfs = []
        for i in range(len(self.metric_names)):
            result_columns = [self.metric_names[i] + '_' + dataset_column for dataset_column in self.dataset_columns]

            best_result_data = np.concatenate([best_result[i], [mean_best_result[i]]], axis=0)[np.newaxis, :]
            best_result_df = pd.DataFrame(data=best_result_data,
                                          columns=result_columns + ['mean' + '_' + self.metric_names[i]])

            th_file = os.path.join(self.log_folder, 'th_{}_{}.npy'.format(self.metric_names[i], self.n_res))
            np.save(th_file,
                    best_thresholds[i])
            th_df = pd.DataFrame(data=[th_file], columns=['th' + '%' + 'mean' + '_' + self.metric_names[i]])
           
            result_file = os.path.join(self.log_folder, '{}_full_{}.csv'.format(self.metric_names[i],
                                                                                self.n_res))
            result_df = pd.DataFrame(data=np.concatenate([thresholds[:, np.newaxis], results[i]], axis=1), columns=['thresholds'] + result_columns)
            result_df = result_df.set_index('thresholds')
            result_df.to_csv(result_file, index=True)
            result_df = pd.DataFrame(data=[result_file], columns=[self.metric_names[i] + '_' + 'full'])

            metric_dfs.append(pd.concat([best_result_df, th_df, result_df], axis=1))
        self.n_res += 1

        return pd.concat(metric_dfs, axis=1)

    def get_best_thresholds(self):
        return self.best_thresholds


class MaskedTestClassificationScores(MaskedClassificationScores):
    def __init__(self, metric_names, pred_in, dataset_columns, thresholds=None, **kwargs):
        if thresholds is None:
            thresholds = (0.5) * tf.ones((len(metric_names), len(dataset_columns)))

        super(MaskedTestClassificationScores, self).__init__(metric_names=metric_names,
                                                             thresholds=tf.expand_dims(thresholds, axis=-1),
                                                             pred_in=pred_in,
                                                             dataset_columns=dataset_columns)
    
    def set_thresholds(self, thresholds):
        self.thresholds.assign(tf.expand_dims(thresholds, axis=-1))


    def result(self):
        result_list = []
        for i in range(len(self.metric_names)):
            result_list.append(tf.expand_dims(self.classification_functions[i](self.tp[i],
                                                                               self.tn[i],
                                                                               self.fp[i],
                                                                               self.fn[i]),
                                              axis=0))
        # (M, N, 1) 
        results = tf.concat(result_list, axis=0)
        return results

    def result_to_df(self):
        # (M, 1, N)
        results = tf.transpose(self.result(), perm=(0, 2, 1)).numpy()
        results_df = []
        for i in range(len(self.metric_names)):
            columns = [self.metric_names[i] + dataset_column for dataset_column in self.dataset_columns] + ['mean_' + self.metric_names[i]]
            results_df.append(pd.DataFrame(data=np.concatenate([results[i], np.atleast_2d(np.mean(results[i]))],
                                                               axis=1),
                                           columns=columns))
        return pd.concat(results_df, axis=1)


class MaskedABAW3ClassificationScores(MaskedTestClassificationScores):
    def __init__(self, metric_names, pred_in, dataset_columns, **kwargs):
        super(MaskedABAW3ClassificationScores, self).__init__(metric_names=metric_names,
                                                              pred_in=pred_in,
                                                              dataset_columns=dataset_columns)

    def result_to_df(self):
        # (M, 1, N)
        results = tf.transpose(self.result(), perm=(0, 2, 1)).numpy()
        results_df = []
        for i in range(len(self.metric_names)):
            columns = ['static_' + self.metric_names[i] + dataset_column for dataset_column in self.dataset_columns] + ['static_' + 'mean_' + self.metric_names[i]]
            results_df.append(pd.DataFrame(data=np.concatenate([results[i], np.atleast_2d(np.mean(results[i]))],
                                                               axis=1),
                                           columns=columns))
        return pd.concat(results_df, axis=1)


SUPPORTED_MASKED_THRESHOLD_METRICS = {"m_th_train": MaskedTrainClassificationScores,
                                      "m_th_test": MaskedTestClassificationScores,
                                      "m_th_abaw3": MaskedABAW3ClassificationScores}
