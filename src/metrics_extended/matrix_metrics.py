import numpy as np
import pandas as pd
import tensorflow as tf

import measures
import os


from metrics_extended.abstract_metrics import SlidingMeanMetric, TensorSlidingMeanMetric, TensorTrackingMetric

class MeanFrobNormToKOrders(SlidingMeanMetric):
    def __init__(self, soft_orders_in, input_in, orders, **kwargs):
        (super(MeanFrobNormToKOrders, self)
         .__init__(name='frobnormtoorders',
                   eval_function=measures.mean_frob2korders(soft_orders_in=soft_orders_in,
                                                            input_in=input_in,
                                                            orders=orders)))

class MeanFrobNormToKSeparation(SlidingMeanMetric):
    def __init__(self, T, K, **kwargs):
        (super(MeanFrobNormToKSeparation, self)
         .__init__(name='frobnormtoKsep',
                   eval_function=measures.mean_frob2Kseparation(T=T,
                                                                K=K)))

class MeanFrobNormToMat(SlidingMeanMetric):
    def __init__(self, mat, **kwargs):
        (super(MeanFrobNormToMat, self)
         .__init__(name='frobnormtomat',
                   eval_function=measures.mean_frobnorm_to_mat(mat=np.array(mat))))

class MeanFrobNormToIdentity(MeanFrobNormToMat):
    def __init__(self, T, **kwargs):
        (super(MeanFrobNormToIdentity, self)
         .__init__(mat=tf.eye(T)))

class MeanFrobNormToOrder(SlidingMeanMetric):
    def __init__(self, T, **kwargs):
        (super(MeanFrobNormToOrder, self)
         .__init__(name='maofrobnorm',
                   eval_function=measures.maonet_frob2toyo(T)))

class MeanFrobNormToCanonicalOrder(SlidingMeanMetric):
    def __init__(self, T, **kwargs):
        (super(MeanFrobNormToCanonicalOrder, self)
         .__init__(name='maofrobnorm',
                   eval_function=measures.maonet_frob2toycan(T)))

class MeanFrobNormToReverseOrder(SlidingMeanMetric):
    def __init__(self, T, **kwargs):
        (super(MeanFrobNormToReverseOrder, self)
         .__init__(name='maofrobnorm',
                   eval_function=measures.maonet_frob2toyreverse(T)))

class MeanSoftOrderMatrix(TensorSlidingMeanMetric):
    def __init__(self, T, soft_orders_in, log_folder, **kwargs):
        super(MeanSoftOrderMatrix, self).__init__(name='mean_softordermatrix',
                                                  eval_function=measures.mean_softorder_matrix(soft_orders_in),
                                                  shape=(T, T))
        self.log_folder = log_folder
        if not(os.path.exists(self.log_folder)):
            os.makedirs(self.log_folder)


    def result_to_df(self):
        result_matrix = self.result().numpy()
        tf.print(self.log_folder, result_matrix)
        matrix_file = os.path.join(self.log_folder, 'mean_softorder_matrix.npy')
        np.save(matrix_file,
                result_matrix)
        return pd.DataFrame(data=[matrix_file], columns=['mean_softorder_matrix'])

class SoftOrderMatrix(TensorTrackingMetric):
    def __init__(self, T, soft_orders_in, log_folder, **kwargs):
        super(SoftOrderMatrix, self).__init__(tensor_in=soft_orders_in,
                                              shape=(T, T),
                                              log_folder=log_folder)

class EntropySoftOrderMatrix(SlidingMeanMetric):
    def __init__(self, soft_orders_in, name='entropy', **kwargs):
        super(EntropySoftOrderMatrix, self).__init__(name=name + '_' + soft_orders_in,
                                                     eval_function=measures.mean_matrix_entropy(soft_orders_in))

SUPPORTED_MATRIX_METRICS = {"mfrob2order": MeanFrobNormToOrder,
                            "mfrob2can": MeanFrobNormToCanonicalOrder, 
                            "mfrob2rev": MeanFrobNormToReverseOrder,
                            "mfrob2id": MeanFrobNormToIdentity,
                            "mfrob2ksep": MeanFrobNormToKSeparation,
                            "mfrob2korders": MeanFrobNormToKOrders,
                            "mfrob2mat": MeanFrobNormToMat,
                            "mean_softorder": MeanSoftOrderMatrix,
                            "softorder": SoftOrderMatrix,
                            "entropy_softorder": EntropySoftOrderMatrix}
