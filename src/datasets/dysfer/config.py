import os
import numpy as np

DYSFER_PATH = os.path.join('..', 'resources', 'DYSFER')
RAW_IMAGES = os.path.join(DYSFER_PATH, 'raw', 'images')
RAW_LABELS = os.path.join(DYSFER_PATH, 'raw', 'labels')
PREPROCESSED_IMAGES = os.path.join(DYSFER_PATH, 'preprocessed', 'images')
VIDEO_PATH = os.path.join(DYSFER_PATH, 'videos')


AU_numbers = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 24, 25, 26, 28, 29, 31, 32, 33, 35, 38, 43])
AU_mapping = dict()
for i in range(len(AU_numbers)):
    AU_mapping['AU{}'.format(AU_numbers[i])] = i

DYSFER_PROJECTION = np.array([0, 1, 3, 2, 28])
