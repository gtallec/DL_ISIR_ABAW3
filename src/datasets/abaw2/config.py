import os

ABAW2_PATH = os.path.join('..', 'resources', 'ABAW2')
PREPROCESSED_FOLD_TEMPLATE = os.path.join(ABAW2_PATH, 'preprocessed', '{}_{}.csv') 
AU_order = ["AU{}".format(au) for au in [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]]
EXPR_order = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

