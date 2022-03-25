import tensorflow as tf
from metrics_extended.abstract_metrics import DatasetMetric
from metrics_extended.aggregated_metrics import MeanMetric
from metrics_extended.classification_metrics import AUCByCoord

class MaskedAUC(DatasetMetric):
    def __init__(self, num_thresholds, n_coords, curve, dataset_columns, pred_in, name='AUC', **kwargs):
        (super(MaskedAUC, self)
         .__init__(dataset_columns=dataset_columns,
                   name=name + '_' + curve,
                   **kwargs))
        self.pred_in = pred_in
        self.aucs = [AUCByCoord(num_thresholds, i, curve=curve) for i in range(n_coords)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # If y_true is not 0 or 1 AUC computation will fail:
        y_true = tf.round(y_true)
        y_dict = y_pred[self.pred_in]
        # (B, S, T)
        pred = y_dict['prediction']
        pred_shape = tf.shape(pred)
        # (B x S, T)
        pred = tf.reshape(pred, (pred_shape[0] * pred_shape[1], pred_shape[2]))
        # (B x S, T)
        y_true = tf.reshape(y_true, (pred_shape[0] * pred_shape[1], pred_shape[2]))
        # (B, S)
        mask = y_dict['padding_mask']
        # (B x S, )
        mask = tf.dtypes.cast(tf.reshape(mask, (-1, )) == 1, tf.bool)
        # (K, )
        remaining_elements = tf.boolean_mask(tensor=tf.range(0, pred_shape[0] * pred_shape[1]), mask=mask)
        y_true_masked = tf.gather(y_true, remaining_elements, axis=0)
        pred_masked = tf.gather(pred, remaining_elements, axis=0)

        for i in range(len(self.aucs)):
            self.aucs[i].update_state(y_true=y_true_masked,
                                      y_pred=pred_masked,
                                      sample_weight=sample_weight)

    def reset_states(self):
        for i in range(len(self.aucs)):
            self.aucs[i].reset_states()

    def result(self):
        return tf.stack([self.aucs[i].result() for i in range(len(self.aucs))])

class MaskedAUC_ROC(MaskedAUC):
    def __init__(self, num_thresholds, n_coords, dataset_columns, pred_in, **kwargs):
        (super(MaskedAUC_ROC, self)
         .__init__(num_thresholds=num_thresholds,
                   n_coords=n_coords,
                   curve='ROC',
                   dataset_columns=dataset_columns,
                   pred_in=pred_in))

class MaskedAUC_PR(MaskedAUC):
    def __init__(self, num_thresholds, n_coords, dataset_columns, pred_in, **kwargs):
        (super(MaskedAUC_PR, self)
         .__init__(num_thresholds=num_thresholds,
                   n_coords=n_coords,
                   curve='PR',
                   dataset_columns=dataset_columns,
                   pred_in=pred_in))


SUPPORTED_MASKED_CLASSIFICATION_METRICS = {"m_auc": MaskedAUC,
                                           "m_auc_roc": MaskedAUC_ROC,
                                           "m_auc_pr": MaskedAUC_PR}
