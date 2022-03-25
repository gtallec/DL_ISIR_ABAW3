import os

import tensorflow as tf
import numpy as np

from datasets.abaw3.generation import gen_abaw3
from datasets.abaw3.pandas_interface import columns_abaw3
from metrics_extended.interface import get_metrics
from models.data_processing.interface import DataProcessing
from routines_extended import evaluate_model_on_dataset

from models.interface import get_model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

best_epoch = 8
experiment_path = os.path.join('/home', 'tallec', 'Thesis', 'resources', 'pretrained_weights', 'best_abaw3')
weights_path = os.path.join(experiment_path, 'checkpoints', '{}-epoch'.format(best_epoch))
thresholds_path = os.path.join(experiment_path, 'th_f1score_{}.npy'.format(best_epoch))
model_dict = dict({"main": {"type": "hencreg",
                            "dependencies": ["encoder", "transformer", "decoder"]},
                   "encoder": {
                       "type": "inceptionv3_encoder",
                       "pooling": None,
                       "weights": "imagenet"},
                   "transformer": {
                       "type": "ViT",
                       "num_patches": 64,
                       "patch_size": 1,
                       "temp_xx": 1.0,
                       "d_model": 128,
                       "mlp_scale": 4,
                       "num_layers": 1,
                       "num_heads": 8,
                       "rate": 0.1},
                   "decoder": {
                       "type": "ttat",
                       "mlp_scale": 4,
                       "d_model": 128,
                       "num_layers": 2,
                       "num_heads": 8,
                       "temp_tt": 1.0,
                       "temp_tx": 1.0,
                       "rate": 0.1,
                       "T": 12}})
model = get_model(model_dict)
model.build((None, 299, 299, 3))
model.load_weights(ckpt_path=weights_path,
                   block='main')

thresholds = tf.constant(np.tile(np.load(thresholds_path)[np.newaxis, :], (6, 1)), dtype=tf.float32)

metrics_dict = [{"type": "th_test",
                 "metric_names": ["f1score", "accuracy", "tp", "tn", "fp", "fn"],
                 "threshold_step": 0.005,
                 "thresholds": thresholds,
                 "pred_in": "global_pred",
                 "n_coords": 12},
                {"type": "auc_roc",
                 "num_thresholds": 200,
                 "pred_in": "global_pred",
                 "n_coords": 12}]

data_processing = DataProcessing([{"type": "resize", "im_h": 299, "im_w": 299}])
metrics = get_metrics(metrics_dict, dataset_columns=columns_abaw3(), log_folder='test')

abaw3 = gen_abaw3(mode='valid', batchsize=64, meta=False, subsample=100)

result_df = evaluate_model_on_dataset(model=model,
                                      data_processing=data_processing,
                                      dataset=abaw3,
                                      metrics=metrics)
f1score_df = result_df.filter(regex='^f1score')
print(f1score_df.mean(axis=1))
