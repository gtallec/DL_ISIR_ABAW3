from datasets.celeba.config import LABELS_NAME, get_projection
from visualisation_utils import trunc
import numpy as np

def visu_celeba(projection_mode='all', stat=False, **kwargs):
    projection = get_projection(projection_mode)
    labels_name = np.array(LABELS_NAME)[projection]

    metric_separator = 50 * '-'
    if not stat:
        def visualisation(log):
            print(metric_separator)
            for metric in log:
                print(metric.upper(), 'SCORE')
                log_metric = log[metric]
                if 'mean' in metric:
                    print('Mean :', trunc(100 * log_metric, 1))
                else:
                    for i in range(len(labels_name)):
                        print(labels_name[i], ' : ', trunc(100 * log_metric[i], 1))
                print(metric_separator)
        return visualisation

    else:
        def visualisation_stat(log_moment):
            moment_1 = log_moment['moment_1']
            moment_2 = log_moment['moment_2']
            n_repeat = log_moment['n_repeat']

            print(metric_separator)
            for metric in moment_1:
                print(metric.upper(), 'SCORE')
                metric_mean = moment_1[metric]
                metric_std = np.sqrt((moment_2[metric] - np.power(metric_mean, 2)))
                if 'mean' in metric:
                    print('Mean : ', trunc(100 * metric_mean, 1), '+-', trunc(100 * metric_std/np.sqrt(n_repeat), 1))
                else:
                    for i in range(len(labels_name)):
                        print(labels_name[i], ' : ',
                              trunc(100 * metric_mean[i], 1), '+-', trunc(100 * metric_std[i]/np.sqrt(n_repeat), 1))
                print(metric_separator)
        return visualisation_stat

