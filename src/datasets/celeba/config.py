import os
import numpy as np

CELEBA_PATH = os.path.join('..', 'resources', 'CelebA')

RAW_ATTRIBUTES = os.path.join(CELEBA_PATH, 'raw', 'labels', 'list_attr_celeba.txt')
RAW_EVAL_STATUS = os.path.join(CELEBA_PATH, 'raw', 'labels', 'list_eval_partition.txt')
RAW_IMAGES = os.path.join(CELEBA_PATH, 'raw', 'images')

PREPROCESSED_IMAGES = os.path.join(CELEBA_PATH, 'preprocessed', 'images')
PREPROCESSED_META_TEMPLATE = os.path.join(CELEBA_PATH, 'preprocessed', '{}.csv')
LABELS_NAME = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
               "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
               "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
               "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
               "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
               "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
               "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
               "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]


ACCESSORY_PROJECTION = [38, 37, 36, 35, 34, 15]
HAIRCUT_PROJECTION = [4, 8, 9, 11, 17]

SUPPORTED_PROJECTIONS = {'all': [i for i in range(40)],
                         'accessory': [15, 34, 35, 37, 38],
                         'haircut': [4, 8, 9, 11, 17],
                         'gender': [18, 20, 22, 24, 36],
                         'misc': [0, 27, 21, 25, 39],
                         'instagram': [1, 2, 19, 29, 25]}

def get_projection(projection_mode):
    return SUPPORTED_PROJECTIONS[projection_mode]

print(np.array(LABELS_NAME)[get_projection('gender')])
print(np.array(LABELS_NAME)[get_projection('instagram')])
print(np.array(LABELS_NAME)[get_projection('accessory')])
print(np.array(LABELS_NAME)[get_projection('misc')])
