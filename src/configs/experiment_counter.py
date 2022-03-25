from configs.config import EXP_COUNT_PATH

def get_experiment_id():
    with open(EXP_COUNT_PATH, 'r') as f:
        experiment_id = int(f.read())
    with open(EXP_COUNT_PATH, 'w') as f:
        f.write(str(experiment_id + 1))
    return experiment_id



