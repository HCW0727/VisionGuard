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

from run import run_CGI_STEREO,run_disparity2bev, run_visual_odometry

parser = argparse.ArgumentParser(description='(CGI-Stereo) full_process.py')
parser.add_argument('--mode', default='webcam', help='select a mode', choices=['image','webcam'])
parser.add_argument('--left_image_path', default='/Users/huhchaewon/Datasets/00/image_2/000001.png', help='left_image path')
parser.add_argument('--right_image_path', default='/Users/huhchaewon/Datasets/00/image_3/000001.png', help='right_image path')

args = parser.parse_args()


if args.mode == 'image':
    root_dir_l = '/Users/huhchaewon/Datasets/00/image_2/'
    root_dir_r = '/Users/huhchaewon/Datasets/00/image_3/'

    file_list = os.listdir(root_dir_l)
    file_list.sort()

    
    start_time = time.time()

    file_name = args.left_image_path.split('/')[-1]
    

    print(args.left_image_path)
    left_image = cv2.imread(args.left_image_path)
    right_image = cv2.imread(args.right_image_path)

    disp,left,right = run_CGI_STEREO.run_model(left_image,right_image)

    
    


    cv2.imwrite('./output/image/disp/'+file_name,disp)
    cv2.imwrite('./output/image/left/'+file_name,left)
    cv2.imwrite('./output/image/right/'+file_name,right)

    run_disparity2bev.disp2pcd(disp,'./output/image/PCD/point_cloud.pcd')

    print('Total time : ',time.time() - start_time)

    cv2.imshow('disparity',disp)
    cv2.waitKey(0)

elif args.mode == 'webcam':
    webcam = cv2.VideoCapture(0)
    middle_w = int(webcam.get(3)/2)
    previous_left = None

    num = 1
    while webcam.isOpened():
        
        start_time = time.time()
        status,frame = webcam.read()
        left_image = frame[:,middle_w:]
        right_image = frame[:,:middle_w]

        left, right,disp = run_CGI_STEREO.run_model(left_image,right_image)

        # if previous_left == None:
        #     previous_left = left

        cv2.imshow('left',left)
        cv2.imshow('right',right)
        cv2.imshow('disparity',disp*2)

        k = cv2.waitKey(100)

        if k == 27:
            break

        cv2.imwrite('./output/webcam/disp/disp'+str(num)+'.png',disp)
        cv2.imwrite('./output/webcam/left/left'+str(num)+'.png',left)
        cv2.imwrite('./output/webcam/right/right'+str(num)+'.png',right)

        # run_disparity2bev.disp2pcd(disp,'./output/webcam/PCD/point_cloud'+str(num)+'.pcd')

        num += 1
        print('Total time : ',time.time() - start_time)



