import os

ABAW3 = os.path.join('..', 'resources', 'ABAW3')
AU_ORDER = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
CSV_TEMPLATE = os.path.join(ABAW3, 'preprocessed', '{}.csv')

VIDEO_CSV_TEMPLATE = os.path.join(ABAW3, 'preprocessed', 'videos', '{}.csv')
VIDEO_TENSOR_TEMPLATE = os.path.join(ABAW3, 'preprocessed', 'videos_tensors', '{}.txt')
SANITY_VIDEO_CSV_TEMPLATE = os.path.join(ABAW3, 'preprocessed', 'sanity_videos', '{}.csv')
VIDEO_SUMMARY_TEMPLATE = os.path.join(ABAW3, 'preprocessed', 'video_summary', '{}.npy')

LABEL_PADDING = os.path.join(ABAW3, 'preprocessed', 'padding', 'labels.txt')
IMG_PADDING = os.path.join(ABAW3, 'preprocessed', 'padding', 'img.jpg')

TRAIN_ABAW3 = os.path.join(ABAW3, 'preprocessed', 'train_for_abaw3.csv')
VALID_ABAW3 = os.path.join(ABAW3, 'preprocessed', 'valid_for_abaw3.csv')
TRAINVAL_ABAW3 = os.path.join(ABAW3, 'preprocessed', 'trainval_for_abaw3.csv')
