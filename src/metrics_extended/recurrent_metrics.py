from metrics_extended.abstract_metrics import SlidingMeanMetric
import measures

class BatchAndTimestepMean(SlidingMeanMetric):
    def __init__(self, **kwargs):
        (super(BatchAndTimestepMean, self)
         .__init__(name="batch_and_timestep_mean",
                   eval_function=measures.mean_by_timestep_and_batch))

class TimestepMetric(SlidingMeanMetric):
    def __init__(self, timestep, **kwargs):
        (super(TimestepMetric, self)
         .__init__(name='timestep',
                   eval_function=measures.timestep_loss(timestep)))


SUPPORTED_RECURRENT_METRICS = {'timestep': TimestepMetric}
