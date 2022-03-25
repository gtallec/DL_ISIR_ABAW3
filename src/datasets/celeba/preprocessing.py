from datasets.celeba.config import RAW_ATTRIBUTES, RAW_EVAL_STATUS, RAW_IMAGES, PREPROCESSED_IMAGES, PREPROCESSED_META_TEMPLATE
from tqdm import tqdm

import pandas as pd
import cv2
import os

def preprocess_labels():
    eval_status_df = pd.read_csv(RAW_EVAL_STATUS,
                                 delim_whitespace=True,
                                 names=['img_id', 'eval_status'])
    eval_status_df = eval_status_df.astype({'img_id': str})

    attribute_df = pd.read_csv(RAW_ATTRIBUTES,
                               delim_whitespace=True)
    attribute_columns = list(attribute_df.columns)
    attribute_columns = attribute_columns.remove('img_id')
    attribute_df[attribute_columns] = (attribute_df[attribute_columns] + 1)/2

    attribute_eval_status = attribute_df.merge(eval_status_df)


    train_attribute_eval_status = attribute_eval_status[attribute_eval_status.eval_status == 0]
    train_attribute_eval_status = train_attribute_eval_status.drop(labels='eval_status', axis=1)
    train_attribute_eval_status.to_csv(PREPROCESSED_META_TEMPLATE.format('train'),
                                       index=False)

    valid_attribute_eval_status = attribute_eval_status[attribute_eval_status.eval_status == 1]
    valid_attribute_eval_status = valid_attribute_eval_status.drop(labels='eval_status', axis=1)
    valid_attribute_eval_status.to_csv(PREPROCESSED_META_TEMPLATE.format('valid'),
                                       index=False)

    test_attribute_eval_status = attribute_eval_status[attribute_eval_status.eval_status == 2]
    test_attribute_eval_status = test_attribute_eval_status.drop(labels='eval_status', axis=1)
    test_attribute_eval_status.to_csv(PREPROCESSED_META_TEMPLATE.format('test'),
                                      index=False)
    return True

def preprocess_image(img_path, show=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img = cv2.resize(img, (299, 299)) 
    return img

def preprocess_images():
    images = os.listdir(RAW_IMAGES)
    success = True
    for image in tqdm(images):
        img_src = os.path.join(RAW_IMAGES, image)
        img_dest = os.path.join(PREPROCESSED_IMAGES, image)
        preprocessed_image = preprocess_image(img_src)
        success = cv2.imwrite(img_dest, preprocessed_image) and success
    return success

def preprocess_celeba():
    return preprocess_images() and preprocess_labels
