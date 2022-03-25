import tensorflow as tf
from metrics_extended.abstract_metrics import DatasetMetric, ScalarTrackingMetric


class Accuracy(DatasetMetric):
    def __init__(self, n_classes, dataset_columns, pred_in, name='categorical_accuracy', **kwargs):
        super(Accuracy, self).__init__(dataset_columns=dataset_columns,
                                       name=name)
        self.pred_in = pred_in
        self.T = len(n_classes)
        self.count = self.add_weight(name='count',
                                     initializer='zeros',
                                     dtype=tf.float32)
        self.accurate = self.add_weight(name='accurate',
                                        initializer='zeros',
                                        shape=(self.T, ))
        task_mask = []
        for i in range(len(n_classes)):
            task_mask_i = tf.concat([tf.zeros((i, n_classes[i])),
                                     tf.ones((1, n_classes[i])),
                                     tf.zeros((self.T - (i + 1), n_classes[i]))],
                                    axis=0)
            task_mask.append(task_mask_i)
        self.task_mask = tf.concat(task_mask, axis=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]
        self.count.assign_add(tf.dtypes.cast(batch_size, tf.float32))

        # (B, T, T_tot)
        task_mask = tf.tile(tf.expand_dims(self.task_mask, axis=0), (batch_size, 1, 1))
        y_pred_tiled = tf.tile(tf.expand_dims(y_pred[self.pred_in], axis=1), (1, self.T, 1)) * task_mask
        y_true_tiled = tf.tile(tf.expand_dims(y_true, axis=1), (1, self.T, 1)) * task_mask

        # (T, )
        accurate_for_batch = tf.math.reduce_sum(tf.math.reduce_sum(y_true_tiled * y_pred_tiled, axis=-1), axis=0)
        self.accurate.assign_add(accurate_for_batch)
        tf.print('count : ', self.count)
        tf.print('accurate : ', self.accurate)

    def reset_states(self):
        self.count.assign(0)
        self.accurate.assign(tf.zeros((self.T, ),
                                      dtype=tf.float32))

    def result(self):
        return self.accurate / self.count

class MeanAccuracy(ScalarTrackingMetric):
    def __init__(self, n_classes, pred_in, name='mean_accuracy', **kwargs):
        super(MeanAccuracy, self).__init__(name=name)
        self.pred_in = pred_in
        self.T = len(n_classes)
        self.count = self.add_weight(name='count',
                                     initializer='zeros',
                                     dtype=tf.float32)
        self.accurate = self.add_weight(name='accurate',
                                        initializer='zeros',
                                        shape=(self.T, ))
        task_mask = []
        for i in range(len(n_classes)):
            task_mask_i = tf.concat([tf.zeros((i, n_classes[i])),
                                     tf.ones((1, n_classes[i])),
                                     tf.zeros((self.T - (i + 1), n_classes[i]))],
                                    axis=0)
            task_mask.append(task_mask_i)
        self.task_mask = tf.concat(task_mask, axis=1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]
        self.count.assign_add(tf.dtypes.cast(batch_size, tf.float32))

        # (B, T, T_tot)
        task_mask = tf.tile(tf.expand_dims(self.task_mask, axis=0), (batch_size, 1, 1))
        y_pred_tiled = tf.tile(tf.expand_dims(y_pred[self.pred_in], axis=1), (1, self.T, 1)) * task_mask
        y_true_tiled = tf.tile(tf.expand_dims(y_true, axis=1), (1, self.T, 1)) * task_mask

        # (T, )
        accurate_for_batch = tf.math.reduce_sum(tf.math.reduce_sum(y_true_tiled * y_pred_tiled, axis=-1), axis=0)
        self.accurate.assign_add(accurate_for_batch)

    def reset_states(self):
        self.count.assign(0)
        self.accurate.assign(tf.zeros((self.T, ),
                                      dtype=tf.float32))

    def result(self):
        return tf.math.reduce_mean(self.accurate) / self.count



SUPPORTED_CATEGORICAL_CLASSIFICATION_METRICS = {"cat_accuracy": Accuracy,
                                                "mean_cat_accuracy": MeanAccuracy}

if __name__ == '__main__':
    accuracy = Accuracy(n_classes=[2, 2, 3])
    y_true = tf.constant([[1, 0, 0, 1, 0, 0, 1]], dtype=tf.float32)
    y_pred = tf.constant([[1, 0, 1, 0, 0, 0, 1]], dtype=tf.float32)
    accuracy.update_state(y_true=y_true,
                          y_pred=y_pred)
    tf.print(accuracy.result())








