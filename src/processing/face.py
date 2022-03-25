from mtcnn import MTCNN
from tqdm import tqdm
from PIL import Image

import numpy as np
import os
import imutils
import dlib
import cv2

def face_detector():
    detector = MTCNN()

    def face_detection(image):
        im = np.array(image)
        return detector.detect_faces(im)
    return face_detection

def crop_image(image, box, padding):
    x, y, width, height = box
    max_y, max_x = image.size
    width_padding = width * padding[0]
    height_padding = height * padding[1]
    
    # Add Forehead contrib to height padding
    crop_min_y = np.maximum(0, np.floor(y - 1.5 * height_padding)).astype(int)
    crop_max_y = np.minimum(max_y, np.ceil(y + height + height_padding)).astype(int)

    crop_min_x = np.maximum(0, np.floor(x - width_padding)).astype(int)
    crop_max_x = np.minimum(max_x, np.ceil(x + width + width_padding)).astype(int)

    return image.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))


def resize(image, target_size):
    return image.resize(target_size)

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def landmark_detector(face_detector, landmark_predictor):
    def detect_landmarks(image):
        rects = face_detector(image, 1)
        # if len(rects) < 1:
            #return False, None 
        # rect = rects[0]
        rect = (0, 160, 160, 160)
        shape = shape_to_np(landmark_predictor(image, rect))
        return True, shape
    return detect_landmarks

def video_face_crop(video, video_src_path, video_dest_path, padding, target_size):
    """ For a given <video> represented as an ordered list of frames :
    Crop face on each frame and save the resulting images in <video_dest_path>.
    """
    box = None
    face_detection = face_detector()
    for frame in tqdm(video):
        image = Image.open(os.path.join(video_src_path, frame))
        face_detections = face_detection(image)
        if len(face_detections) != 0:
            box = face_detections[0]['box']
        cropped_image = crop_image(image, box, padding=padding)
        resized_cropped_image = resize(cropped_image, target_size)
        resized_cropped_image.save(os.path.join(video_dest_path, frame), format='JPEG')

def video_face_crop(video, video_src_path, video_dest_path, padding, target_size):
    """ For a given <video> represented as an ordered list of frames :
    Crop face on each frame and save the resulting images in <video_dest_path>.
    """
    box = None
    face_detection = face_detector()
    for frame in tqdm(video):
        image = Image.open(os.path.join(video_src_path, frame))
        face_detections = face_detection(image)
        if len(face_detections) != 0:
            box = face_detections[0]['box']
        cropped_image = crop_image(image, box, padding=padding)
        resized_cropped_image = resize(cropped_image, target_size)
        resized_cropped_image.save(os.path.join(video_dest_path, frame), format='JPEG')

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from PIL import Image

    img_src = os.path.join('..', '..', 'resources', 'test.jpg')
    img = np.array(Image.open(img_src))

    detector = MTCNN()
    box = face_detector(detector)(img)[0]['box']
    print(box)
    cropped_image = crop_image(img, box, (0.2, 0.2))
    print(cropped_image.shape)
    plt.imshow(img)
    plt.show()


