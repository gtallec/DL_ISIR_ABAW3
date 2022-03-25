import processing.face
import pandas as pd
import os
from datasets.disfa.config import AU_ORDER, PREPROCESSED_AU_TEMPLATE, VIDEO_NUMBERS, PREPROCESSED_META, RAW_AU_TEMPLATE, fold_assignment, JAANET_FOLDS, PREPROCESSED_VIDEO_TEMPLATE, FRAME_TEMPLATE, JAANET_path

"""
def preprocess_video_labels(video):
    lines_by_file = []
    video_raw_au_template = RAW_AU_TEMPLATE.format(video, '{}')
    for au in AU_ORDER:
        print(video_raw_au_template.format(au))
        with open(video_raw_au_template.format(au)) as f:
            lines_by_file.append(f.read().splitlines())

    with open(PREPROCESSED_AU_TEMPLATE.format(video)):
        n_frames = len(lines_by_file[0])
        for i in range(1, n_frames + 1):
            line = str(i)
            for lines in lines_by_file:
                line = line + ',' + lines[i-1].split(',')[1]
            line = line + '\n'

def labels_to_csv(assignement_type='classic'):
    fold_assignment_dict = fold_assignment(assignment_type=assignement_type)
    columns = ['frame'] + ['AU{}'.format(au) for au in AU_ORDER]
    dfs = []
    for video_number in VIDEO_NUMBERS:
        print(video_number)
        video_df = pd.read_csv(header=None,
                               names=columns,
                               filepath_or_buffer=PREPROCESSED_AU_TEMPLATE.format(video_number))
        video_df['video'] = video_number
        video_df['fold'] = fold_assignment_dict[video_number]
        dfs.append(video_df)
     
    df = pd.concat(dfs)
    df.to_csv(path_or_buf=PREPROCESSED_META.format(assignement_type),
              index=False,
              columns=columns + ['video', 'fold'])
"""

def labels_to_csv():
    # fold_assignment_dict = fold_assignment(assignment_type=assignement_type)
    columns = ['frame'] + ['AU{}'.format(au) for au in AU_ORDER]
    dfs = []
    for video_number in VIDEO_NUMBERS:
        print(video_number)
        video_df = pd.read_csv(header=None,
                               names=columns,
                               filepath_or_buffer=PREPROCESSED_AU_TEMPLATE.format(video_number)
                              )
        video_df['video'] = video_number
        dfs.append(video_df)
     
    df = pd.concat(dfs)
    # df.to_csv(path_or_buf=PREPROCESSED_META.format(assignement_type),
              # index=False,
              # columns=columns + ['video', 'fold'])
    return df

def jaanet_fold_csv():
    full_csv = labels_to_csv()
    folds_csv = []
    
    for i in range(len(JAANET_FOLDS)):
        file_csv = pd.read_csv(filepath_or_buffer=JAANET_FOLDS[i], sep='/', header=None, names=['ds', 'video', 'frame'])
        file_csv['video'] = file_csv['video'].map(lambda x: x[2:])
        file_csv['frame'] = file_csv['frame'].map(lambda x: int(x[:-4]) + 1)
        file_csv = file_csv.drop(['ds'], axis=1)
        fold_csv = pd.merge(full_csv, file_csv, on=['frame', 'video'])
        fold_csv['path'] = fold_csv.apply(lambda x: os.path.join(PREPROCESSED_VIDEO_TEMPLATE + x['video'],
                                                                 FRAME_TEMPLATE + str(x['frame']) + '.jpeg'), axis=1)
        
        fold_csv = fold_csv.drop(['video', 'frame'], axis=1)
        folds_csv.append(fold_csv)
        
        fold_csv.to_csv(path_or_buf=os.path.join(JAANET_path, 'fold{}.csv'.format(i)),
                        index=False,
                        header=True)
        
    fold_01 = pd.concat([folds_csv[0], folds_csv[1]], ignore_index=True)
    fold_01.to_csv(path_or_buf=os.path.join(JAANET_path, 'fold01.csv'),
                   index=False,
                   header=True)
    fold_02 = pd.concat([folds_csv[0], folds_csv[2]], ignore_index=True)
    fold_02.to_csv(path_or_buf=os.path.join(JAANET_path, 'fold02.csv'),
                   index=False,
                   header=True)
    fold_12 = pd.concat([folds_csv[1], folds_csv[2]], ignore_index=True)
    fold_12.to_csv(path_or_buf=os.path.join(JAANET_path, 'fold12.csv'),
                   index=False,
                   header=True)
         
    return fold_csv
