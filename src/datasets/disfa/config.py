import os

DISFA = os.path.join('..', 'resources', 'DISFA')

RAW_VIDEO_TEMPLATE = os.path.join(DISFA, 'FRAMES', 'SN{}')
PREPROCESSED_VIDEO_TEMPLATE = os.path.join(DISFA, 'preprocessed', 'images', 'SN')
PREPROCESSED_META = os.path.join(DISFA, 'preprocessed', 'preprocessed_{}.csv')
FRAMES_PER_VIDEO = os.path.join(DISFA, 'preprocessed', '299x299', 'videos', '{}.csv')
FRAME_TEMPLATE = 'frame'

RAW_AU_TEMPLATE = os.path.join(DISFA, 'ActionUnit_Labels', 'SN{}', 'au{}.txt')
PREPROCESSED_AU_TEMPLATE = os.path.join(DISFA, 'preprocessed', 'labels', 'SN{}.txt')

JAANET_path = os.path.join(DISFA, 'preprocessed', 'jaanet_folds')
JAANET_FOLDS = [os.path.join(JAANET_path, 'DISFA_part1_path.txt'),
                os.path.join(JAANET_path, 'DISFA_part2_path.txt'),
                os.path.join(JAANET_path, 'DISFA_part3_path.txt')]
FOLD_TEMPLATE = os.path.join(DISFA, 'preprocessed', '{}', '{}_folds', '{}.csv') 
AU_ORDER = [12, 17, 20, 26, 4, 6, 15, 1, 25, 2, 5, 9]
SOTA_PROJECTION = [7, 9, 4, 5, 11, 0, 8, 3] # 10, 6, 1, 2]
DYSFER_PROJECTION = [7, 9, 4, 10]
SOTA_ORDER = [1, 2, 4, 6, 9, 12, 25, 26, 5, 15, 17, 20]

VIDEO_NUMBERS = ["001", "002", "003", "004", "005", "006", "007", "008", "009",
                 "010", "011", "012", "013", "016", "017", "018", "021", "023",
                 "024", "025", "026", "027", "028", "029", "030", "031", "032"]


jaanet_video_per_fold = {"0": ['SN001', 'SN002', 'SN009', 'SN010', 'SN016', 'SN026', 'SN027', 'SN030', 'SN032'],
                         "1": ['SN006', 'SN011', 'SN012', 'SN013', 'SN018', 'SN021', 'SN024', 'SN028', 'SN031'],
                         "2": ['SN003', 'SN004', 'SN005', 'SN007', 'SN008', 'SN017', 'SN023', 'SN025', 'SN029']}

classic_video_per_fold = {"0": ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009'],
                          "1": ['SN010', 'SN011', 'SN012', 'SN013', 'SN016', 'SN017', 'SN018', 'SN021', 'SN023'],
                          "2": ['SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030', 'SN031', 'SN032']}

custom_video_per_fold = {"0": ['SN006', 'SN011', 'SN012', 'SN021', 'SN031', 'SN009', 'SN030', 'SN004', 'SN008'],
                         "1": ['SN013', 'SN018', 'SN024', 'SN028', 'SN001', 'SN026', 'SN032', 'SN023', 'SN029'],
                         "2": ['SN002', 'SN010', 'SN016', 'SN027', 'SN003', 'SN005', 'SN007', 'SN017', 'SN025']}


SUPPORTED_FOLD_REPARTITIONS = {'classic': classic_video_per_fold,
                               'jaanet': jaanet_video_per_fold,
                               'custom': custom_video_per_fold}

SUPPORTED_PROJECTIONS = {"dysfer": DYSFER_PROJECTION,
                         "sota": SOTA_PROJECTION}


DISFA_FOR_ABAW3 = os.path.join(DISFA, "preprocessed", "disfa_for_abaw3.csv")


def fold_repartition_list(fold_repartition):
    return SUPPORTED_FOLD_REPARTITIONS.get(fold_repartition, jaanet_video_per_fold)

def get_projection(projection_mode):
    return SUPPORTED_PROJECTIONS.get(projection_mode, "sota") 

