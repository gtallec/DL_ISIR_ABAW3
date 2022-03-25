import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pandas as pd
import sys
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datasets.abaw3.config import AU_ORDER


video_frame_counts_path = os.path.join('..', 'resources', 'ABAW3', 'preprocessed', 'AU_ABAW_Challenge_test_files_expected_frames.txt')

frame_count_df = pd.read_csv(video_frame_counts_path,
                             header=None,
                             names=['video', 'expected_n_frames'])
# To check if submission is correct
def check_submission(submission_dir):
    videos = []
    frames = []
    for video in frame_count_df['video']:
        path_in_submission = os.path.join(submission_dir, video + '.txt')
        print(path_in_submission)
        if not os.path.exists(path_in_submission):
            print('A video is missing')
            return False
        video_df = pd.read_csv(path_in_submission)
        videos.append(video)
        frames.append(video_df.shape[0])
    
    video_name_df = pd.DataFrame(data=videos, columns=['video'])
    frames_df = pd.DataFrame(data=frames, columns=['real_n_frames'])
    video_dfs = pd.concat([video_name_df, frames_df], axis=1)

    full_df = pd.merge(frame_count_df, video_dfs, on='video')
    return full_df[full_df['expected_n_frames'] != full_df['real_n_frames']].shape[0] == 0

# Post Processing Pipeline
def masked_avg_pooling(y_pred, mask, window_size):
    """ y_pred (N, T), mask (B, ) """
    y_pred = tf.constant(y_pred, tf.float32)
    mask = tf.constant(mask, tf.float32)

    T = tf.shape(y_pred)[1]
    # (T, N, 1)
    y_pred = tf.transpose(y_pred)[:, :, tf.newaxis]
    mask = tf.tile(mask[tf.newaxis, :, tf.newaxis], (T, 1, 1))
    filters = tf.ones((window_size, 1, 1))
    # (T, N, 1)
    y_padded = tf.nn.conv1d(input=y_pred,
                            filters=filters,
                            stride=1,
                            padding='SAME')
    n_by_window = tf.nn.conv1d(input=mask,
                               filters=filters,
                               stride=1,
                               padding='SAME')
    return tf.transpose(tf.squeeze(y_padded / (n_by_window + 1e-7), axis=-1)).numpy()

def convolve_df(df, window_size):
    df_columns = df.columns
    df_AU = list(df_columns)[:-1]
    # (N, T)
    np_au = df[df_AU].to_numpy().astype(float)
    # (N, )
    mask = df['mask'].to_numpy().astype(float)
    # (N, T)
    avg_np_au = masked_avg_pooling(np_au, mask, window_size)
    return pd.DataFrame(data=avg_np_au, columns=df_AU)

def confusion_df(pred_df, true_df, columns):
    pred_np = pred_df[columns].to_numpy()
    true_np = true_df[columns].to_numpy()
    tp = np.sum(pred_np * true_np, axis=0)
    fp = np.sum(pred_np * (1 - true_np), axis=0)
    fn = np.sum((1 - pred_np) * true_np, axis=0)
    tn = np.sum((1 - pred_np) * (1 - true_np), axis=0)
    return tp, fp, fn, tn

def confusion_th_df(pred_df, true_df, th, columns):
    """ th (N_t, T) """
    pred_np = pred_df[columns].to_numpy()
    true_np = true_df[columns].to_numpy() 
    # (B, N_t, T)
    pred_bin_np = (pred_np[:, np.newaxis, :] - th[np.newaxis, :, :] >= 0).astype(float)
    tp = np.sum(pred_bin_np * true_np[:, np.newaxis, :], axis=0)
    fp = np.sum(pred_bin_np * (1 - true_np[:, np.newaxis, :]), axis=0) 
    fn = np.sum((1 - pred_bin_np) * true_np[:, np.newaxis, :], axis=0)
    tn = np.sum((1 - pred_bin_np) * (1 - true_np[:, np.newaxis, :]), axis=0)
    return tp, fp, fn, tn

def precision(tp, fp, fn, tn):
    """ all matrices are (..., T) """
    return (tp + 1e-7) / (tp + fp + 1e-7)

def recall(tp, fp, fn, tn):
    """ all matrices are (..., T) """
    return (tp + 1e-7) / (tp + fn + 1e-7)

def f1score(tp, fp, fn, tn):
    prec = precision(tp, fp, fn, tn)
    rec = recall(tp, fp, fn, tn)
    return 2 * prec * rec / (prec + rec + 1e-7)

def best_th(metric_mat, th):
    # th (N_t, T), matrix_metric (N_t, T)
    # (T, )
    best_score = np.max(metric_mat, axis=0)
    T = best_score.shape[0]
    best_mean_score = np.mean(best_score)

    best_score_index = np.argmax(metric_mat, axis=0)
    best_th = th[best_score_index, np.arange(T)]
    return best_score, best_mean_score, best_th


def get_mask(path):
    if 'padding' in path:
        return 0
    else:
        return 1

def process_video_df(pred_df, columns, window_size, face_normalization):
    pred_df['mask'] = pred_df['path'].map(get_mask)
    pred_df = pred_df[columns + ['mask']]
    if window_size is not None:
        pred_df = convolve_df(pred_df, window_size=window_size)
        
    if face_normalization:
        pred_df = normalize_video_df(pred_df, columns=columns)

    return pred_df[columns]

def f1score_videos(video_list,
                   pred_file_template,
                   true_file_template,
                   th,
                   columns,
                   face_normalization=False,
                   window_size=None):
    tps, fps, fns, tns = [], [], [], []
    for video in tqdm(video_list):
        pred_path = pred_file_template.format(video)
        pred_df = pd.read_csv(pred_path)
        pred_df = process_video_df(pred_df=pred_df,
                                   columns=columns,
                                   face_normalization=face_normalization,
                                   window_size=window_size)
        true_path = true_file_template.format(video)
        true_df = pd.read_csv(true_path)[columns]

        tp, fp, fn, tn = confusion_th_df(pred_df=pred_df,
                                         true_df=true_df,
                                         th=th,
                                         columns=columns)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
    tps = np.sum(np.stack(tps, axis=0), axis=0)
    fps = np.sum(np.stack(fps, axis=0), axis=0)
    fns = np.sum(np.stack(fns, axis=0), axis=0)
    tns = np.sum(np.stack(tns, axis=0), axis=0)
    f1score_eval = f1score(tps, fps, fns, tns)
    return best_th(f1score_eval, th)

def th_df(video_df, thresholds, columns):
    video_np = video_df[columns].to_numpy()
    video_np_th = (video_np - thresholds[np.newaxis, :] >= 0).astype(int) 
    return pd.DataFrame(data=video_np_th, columns=columns)


def hypertune_window_size(video_list, pred_file_template, true_file_template, columns, th, window_sizes, face_normalization):
    f1score_by_window_size = []
    thresholds_by_window_size = []
    for window_size in window_sizes:
        best_score, best_mean_score, best_th = f1score_videos(video_list,
                                                              pred_file_template,
                                                              true_file_template,
                                                              th,
                                                              columns,
                                                              window_size=window_size,
                                                              face_normalization=face_normalization)
        thresholds_by_window_size.append(best_th)
        f1score_by_window_size.append(best_mean_score)

    f1score_by_window_size = np.stack(f1score_by_window_size, axis=0)
    thresholds_by_window_size = np.stack(thresholds_by_window_size, axis=0)
    window_sizes = np.array(window_sizes)
    best_f1score_index = np.argmax(f1score_by_window_size)
    return f1score_by_window_size[best_f1score_index], thresholds_by_window_size[best_f1score_index], window_sizes[best_f1score_index]

def normalize_video_df(video_df, columns):
    video_pred_np = video_df[columns].to_numpy()
    video_logit_np = reverse_sigmoid(video_pred_np)
    n_video_logit_np = video_logit_np - np.mean(video_logit_np, axis=0)[np.newaxis, :]
    n_video_pred_np = sigmoid(n_video_logit_np)
    return pd.DataFrame(data=n_video_pred_np, columns=columns)

def prepare_submission(video_list, pred_file_template, submission_file_template, columns, th, window_size, face_normalization=False):
    for video in video_list:
        video_df = pd.read_csv(pred_file_template.format(video))
        preprocessed_video_df = process_video_df(video_df,
                                                 columns,
                                                 window_size,
                                                 face_normalization)
        th_preprocessed_video_df = th_df(preprocessed_video_df,
                                         thresholds=th,
                                         columns=columns)
        th_preprocessed_video_df.to_csv(submission_file_template.format(video.split('.')[0] + '.txt'), index=False)
    submission_check = check_submission(os.path.dirname(submission_file_template))
    return submission_check

def hypertune_and_prepare_submission(valid_folder,
                                     valid_pred_template,
                                     valid_true_template,
                                     test_folder,
                                     test_pred_template,
                                     submission_file_template,
                                     columns,
                                     th,
                                     window_sizes,
                                     face_normalization=False):
    valid_video_list = [video for video in os.listdir(valid_folder) if os.path.splitext(video)[-1] == '.csv']
    test_video_list = [video for video in os.listdir(test_folder) if os.path.splitext(video)[-1] == '.csv']
    best_f1score, best_thresholds, best_window_size = hypertune_window_size(video_list=valid_video_list,
                                                                            pred_file_template=valid_pred_template,
                                                                            true_file_template=valid_true_template,
                                                                            columns=columns,
                                                                            th=th,
                                                                            window_sizes=window_sizes,
                                                                            face_normalization=face_normalization)
    print("VALID F1SCORE : ", best_f1score)
    print('VALID BEST WINDOW : ', best_window_size)
    print('Valid best thresholds : ', best_thresholds)
    return prepare_submission(video_list=test_video_list,
                              pred_file_template=test_pred_template,
                              submission_file_template=submission_file_template,
                              columns=columns,
                              th=best_thresholds,
                              window_size=best_window_size,
                              face_normalization=face_normalization)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reverse_sigmoid(p):
    return np.log(1 / ((1 / p) - 1))


if __name__ == '__main__':
    submissions_folder = os.path.join('..', 'abaw3_submissions')
    valid_folder = os.path.join(submissions_folder, 'submission_2', 'valid_proba')
    valid_pred_template = os.path.join(valid_folder, '{}')
    valid_true_template = os.path.join(submissions_folder, 'valid_videos_gt', '{}')
 
    test_folder =  os.path.join(submissions_folder, 'submission_2', 'test_proba')
    test_pred_template = os.path.join(test_folder, '{}')
    submission_file_template = os.path.join(submissions_folder, 'submission_2', 'final_submission', '{}')
    columns = AU_ORDER
    th = np.tile(np.linspace(0, 1, 200)[:, np.newaxis], (1, 12))
    window_sizes = [5, 10, 15, 20, 25]
    face_normalization=False
    
    hypertune_and_prepare_submission(valid_folder,
                                     valid_pred_template,
                                     valid_true_template,
                                     test_folder,
                                     test_pred_template,
                                     submission_file_template,
                                     columns,
                                     th,
                                     window_sizes,
                                     face_normalization=False)

