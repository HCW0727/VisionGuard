import cv2
import numpy as np

def resize_img(img):
    h, w = img.shape[:2]
    nonzero_coords = np.nonzero(img)
    reduced_coords = (np.array(nonzero_coords) / 3).astype(np.uint8)
    img_reduced = np.zeros((h // 3, w // 3), dtype=np.uint8)
    img_reduced[reduced_coords[0], reduced_coords[1]] = 130

    return img_reduced

def expand_path(path):
    scale = 3
    new_path = []
    for x,y in path:
        new_path.append([(x+1)*scale-1,(y+1)*scale-1])
    return new_path