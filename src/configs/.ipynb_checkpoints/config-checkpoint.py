import os

MACHINE_ID = 2
PROJECT_PATH = os.path.join('/home', 'sdi', 'gtallec', 'Thesis')
LOG_PATH = os.path.join(PROJECT_PATH, 'logs')
RESULT_PATH = os.path.join(PROJECT_PATH, 'results')
PRETRAINED_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'resources', 'pretrained_weights')
EXP_COUNT_PATH = os.path.join(PROJECT_PATH, 'src', 'configs', 'experiment_id.txt')