import Augmentor
import random
import cv2
import numpy as np
from pathlib import Path
import os
from scipy.misc import imread
from preprocess.normalize import preprocess_signature

skew_tilt = [0.5, 0.6, 0.7]

def augment(path):
    p = Augmentor.Pipeline(path)

    for tilt in skew_tilt:
        p = Augmentor.Pipeline(path)
        p.skew_tilt(1, tilt)
        p.skew_corner(1, tilt)
        p.process()

        p = Augmentor.Pipeline(path)
        p.skew_top_bottom(1, tilt)
        p.skew_left_right(1, tilt)
        p.process()


def resize_with_pad(img_path):

    image = cv2.imread(img_path)
    height, width, _ = image.shape

    def get_padding_size(image):
        h, w, _ = image.shape
        top = (int) (0.5*h)
        bottom = (int) (0.5*h)
        left = (int) (0.5*w)
        right = (int) (0.5*w)
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [255, 255, 255]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (width, height))

    padding_folder = os.path.join(Path(img_path).parent, 'padding')
    print(Path(img_path).name)

    os.makedirs(padding_folder, exist_ok=True)
    cv2.imwrite(os.path.join(padding_folder, Path(img_path).name), resized_image)

def scale_image(img_path):
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    resized = cv2.resize(image, (w*2, h))
    padding_folder = os.path.join(Path(img_path).parent, 'scale')

    os.makedirs(padding_folder, exist_ok=True)
    cv2.imwrite(os.path.join(padding_folder, Path(img_path).name), resized)

    return resized

def cut_block(img_path, canvas_size = (952, 1360)):
    original = imread(img_path, flatten=1)
    processed = preprocess_signature(original, canvas_size)
    threshold, binarized_image = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r, c = np.where(binarized_image > 0)
    
    h, w = processed.shape
    kernel_width = random.randint(35,40)
    half = kernel_width//2

    i = 0
    while i!= 1:
        num = random.randrange(0, len(r), half)
        if r[num] - half >= 0 and r[num] + half <= w and c[num] - half >= 0 and c[num] + half <= h:
            i += 1
            processed[r[num] - half: r[num] + half, c[num] - half: c[num] + half] = 0
    
    padding_folder = os.path.join(Path(img_path).parent, 'cut')

    os.makedirs(padding_folder, exist_ok=True)
    cv2.imwrite(os.path.join(padding_folder, Path(img_path).name), processed)
    # cv2.imshow('processed', processed)
    # # cv2.imshow('threshold', threshold)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return processed


if __name__ == "__main__":
    cut_block('data/tu/data_1570086474.png')
    # augment('data/tu/padding')