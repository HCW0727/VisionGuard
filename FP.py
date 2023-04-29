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

f = open('time_stamp.txt','w')


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

    num = 1
    while webcam.isOpened():
        current_time = time.time()
        current_time_rounded = round(current_time, 3)
        time_str = time.strftime("%H:%M:%S", time.localtime(current_time_rounded))
        time_str += "{:.3f}".format(current_time_rounded - int(current_time_rounded))[1:]

        print(time_str)
        f.writelines(time_str + '\n')

        
        start_time = time.time()
        status,frame = webcam.read()
        left_image = frame[:,middle_w:]
        right_image = frame[:,:middle_w]

        # left, right, disp = run_CGI_STEREO.run_model(left_image,right_image)

        # if previous_left == None:
        #     previous_left = left

        cv2.imshow('left',left_image)
        cv2.imshow('right',right_image)
        # cv2.imshow('disparity',disp)

        k = cv2.waitKey(200)

        if k == 27:
            break

        # cv2.imwrite('./output/webcam/disp/disp'+str(num)+'.png',disp)

        cv2.imwrite('./test_set/left/'+str(num)+'.png',left_image)
        cv2.imwrite('./test_set/right/'+str(num)+'.png',right_image)

        

        # run_disparity2bev.disp2pcd(disp,'./output/webcam/PCD/point_cloud'+str(num)+'.pcd')

        num += 1
        # print('Total time : ',time.time() - start_time)



