import os

AU_ORDER = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24"]
SEX_ORDER = ["M", "F"]
ETHNICITY_ORDER = ["AA", "A", "EA", "H"]
TASK_ORDER = ["task{}".format(i) for i in range(1, 9)]

AU_COLUMNS = range(0, 24)
AU_BINARY = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
SEX_COLUMNS = range(24, 26)
ETHNICITY_COLUMNS = range(26, 30)
TASK_COLUMNS = range(30, 38)

BP4D_PATH = os.path.join('..', 'resources', 'BP4D')
BP4D_FOR_ABAW3 = os.path.join(BP4D_PATH, 'preprocessed', 'bp4d_for_abaw3.csv') 
FOLD_TEMPLATE = os.path.join(BP4D_PATH, 'preprocessed', 'folds', '{}.csv') 
