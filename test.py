import cv2
from PIL import Image
from datasets.data_io import get_transform
from models import __models__
import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import time

from run import run_CGI_STEREO,run_disparity2bev,run_global_map
from run import run_visual_odometry, run_pathfinding

def resize_img(img):
    h, w = img.shape[:2]
    nonzero_coords = np.nonzero(img)
    reduced_coords = (np.array(nonzero_coords) / 3).astype(np.uint8)
    img_reduced = np.zeros((h // 3, w // 3), dtype=np.uint8)
    img_reduced[reduced_coords[0], reduced_coords[1]] = 130

    return img_reduced

img1 = cv2.imread('output/global_map/000000.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('output/global_map/000010.png',cv2.IMREAD_GRAYSCALE)


s_start = (15, 187)
s_goal = (127, 15)

img1_resized = resize_img(img1)
img2_resized = resize_img(img2)


img1_draw = img1_resized.copy()
img2_draw = img2_resized.copy()

dstar = run_pathfinding.DStar(img1_resized,s_start,s_goal,"euclidean")
path = dstar.run()

st = time.time()
path = dstar.on_change(img2_resized)
print(time.time()-st)

for px,py in path:
    img2_draw[py,px] = 123

cv2.imshow('test',img2_draw)
cv2.waitKey(0)
# print(path)
