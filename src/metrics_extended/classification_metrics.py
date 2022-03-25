import tensorflow as tf
from metrics_extended.abstract_metrics import DatasetMetric
from metrics_extended.aggregated_metrics import MeanMetric

class AUCByCoord(tf.keras.metrics.Metric):
    def __init__(self, num_thresholds, coord, curve='ROC', **kwargs):
        (super(AUCByCoord, self)
         .__init__(**kwargs))
        self.auc = tf.keras.metrics.AUC(num_thresholds=num_thresholds, curve=curve)
        self.coord = coord

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_coord = y_true[..., self.coord]
        y_true_coord = tf.reshape(y_true_coord, [-1])

        y_pred_coord = y_pred[..., self.coord]
        y_pred_coord = tf.reshape(y_pred_coord, [-1])

        self.auc.update_state(y_true_coord, y_pred_coord)

    def reset_states(self):
        self.auc.reset_states()

    def result(self):
        return self.auc.result()

class AUC(DatasetMetric):
    def __init__(self, num_thresholds, n_coords, curve, dataset_columns, pred_in='pred', name='AUC', **kwargs):
        (super(AUC, self)
         .__init__(dataset_columns=dataset_columns,
                   name=name + '_' + curve,
                   **kwargs))
        self.pred_in = pred_in
        self.aucs = [AUCByCoord(num_thresholds, i, curve=curve) for i in range(n_coords)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle the case in train when y_true comes from augmentations that modifies the labels
        # eg: Label Smoothing/Mixup
        y_true = tf.round(y_true)
        for i in range(len(self.aucs)):
            self.aucs[i].update_state(y_true=y_true,
                                      y_pred=y_pred[self.pred_in],
                                      sample_weight=sample_weight)

    def reset_states(self):
        for i in range(len(self.aucs)):
            self.aucs[i].reset_states()

    def result(self):
        return tf.stack([self.aucs[i].result() for i in range(len(self.aucs))])

class AUC_ROC(AUC):
    def __init__(self, num_thresholds, n_coords, dataset_columns, pred_in, **kwargs):
        (super(AUC_ROC, self)
         .__init__(num_thresholds=num_thresholds,
                   n_coords=n_coords,
                   curve='ROC',
                   dataset_columns=dataset_columns,
                   pred_in=pred_in))

class AUC_PR(AUC):
    def __init__(self, num_thresholds, n_coords, dataset_columns, pred_in, **kwargs):
        (super(AUC_PR, self)
         .__init__(num_thresholds=num_thresholds,
                   n_coords=n_coords,
                   curve='PR',
                   dataset_columns=dataset_columns,
                   pred_in=pred_in))

class MeanAUC(MeanMetric):
    def __init__(self, num_thresholds, n_coords, curve, **kwargs):
        (super(MeanAUC, self)
         .__init__(AUC(num_thresholds=num_thresholds,
                       n_coords=n_coords,
                       curve=curve)))

class MeanAUC_ROC(MeanAUC):
    def __init__(self, num_thresholds, n_coords, **kwargs):
        (super(MeanAUC_ROC, self)
         .__init__(num_thresholds=num_thresholds,
                   n_coords=n_coords,
                   curve='ROC'))

class MeanAUC_PR(MeanAUC):
    def __init__(self, num_thresholds, n_coords, **kwargs):
        (super(MeanAUC_PR, self)
         .__init__(num_thresholds=num_thresholds,
                   n_coords=n_coords,
                   curve='PR'))


SUPPORTED_CLASSIFICATION_METRICS = {"auc": AUC,
                                    "Mauc": MeanAUC,
                                    "auc_roc": AUC_ROC,
                                    "Mauc_roc": MeanAUC_ROC,
                                    "auc_pr": AUC_PR,
                                    "Mauc_pr": MeanAUC_PR}
