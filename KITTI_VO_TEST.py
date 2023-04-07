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
from run import run_visual_odometry, run_pathfinding, run_size



root_dir_l = '/Users/huhchaewon/Datasets/00/image_2/'
root_dir_r = '/Users/huhchaewon/Datasets/00/image_3/'

file_list = os.listdir(root_dir_l)
file_list.sort()

one_px = 0.33

##################################################
#Value

start_point = (45//3, 561//3)
goal_point = (381//3, 45//3)
 
##################################################

first_img = file_list[0]

current_angle = 0

for img_dir in file_list:
    st = time.time()
    ##################################################
    #image input
    start_time = time.time()

    file_num = int(img_dir[:-4])
    file_name = img_dir
    
    left_image = cv2.imread(os.path.join(root_dir_l,img_dir))
    right_image = cv2.imread(os.path.join(root_dir_r,img_dir))

    ##################################################
    #Stereo to Disparity

    left,right,disp = run_CGI_STEREO.run_model(left_image,right_image)
    images_curr = [left,right,disp]

    ##################################################
    #Disparity to BEV

    BEV_image = run_disparity2bev.disp2bev(disp)


    ##################################################
    #BEV to global map (STEREO VISUAL ODOMETRY), PathFinding
    if img_dir == first_img:
        images_prev = images_curr

        global_map_org = cv2.imread('global_map.png',cv2.IMREAD_GRAYSCALE)
        global_map = run_global_map.overlap(global_map_org,BEV_image,0,0,0)

        #Pathfinding (initial calculation)
        global_map_reduced = run_size.resize_img(global_map_org)
        dstar = run_pathfinding.DStar(global_map_reduced,start_point,goal_point,"euclidean")
        path = dstar.run()

    else:
        moved_x,moved_z,rotated_angle = run_visual_odometry.run_VO(images_prev,images_curr)
        rotated_angle *= -1
        images_prev = images_curr

        theta_rad = np.deg2rad(-current_angle)

        rot_matrix = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                            [0, 1, 0],
                            [-np.sin(theta_rad), 0, np.cos(theta_rad)]])

        input_vec = np.array([[moved_x], [0], [moved_z]])
        output_vec = np.dot(rot_matrix, input_vec)

        moved_x,moved_z = output_vec[0][0], output_vec[2][0]

        moved_x,moved_z = moved_x/one_px,moved_z/one_px

        current_angle += rotated_angle

        moved_x,moved_z,current_angle = round(moved_x),round(moved_z),round(current_angle)


        global_map = run_global_map.overlap(global_map,BEV_image,moved_x,moved_z,current_angle)

        ##################################################
        #PathFinding (on_change)

        global_map_reduced = run_size.resize_img(global_map)

        st2 = time.time()
        path = dstar.on_change(global_map_reduced)
        print('PFtime',time.time()-st2)
    
    
    ##################################################
    #image output
    global_map_reduced_draw = global_map_reduced.copy()
    for px,py in path:
        global_map_reduced_draw[py,px] = 255

    global_map_draw = global_map.copy()
    for px,py in run_size.expand_path(path):
        global_map_draw[py,px] = 255

    print('time : ',time.time()-st)
    
    cv2.imwrite('./output/image/disp/'+img_dir,disp)
    cv2.imwrite('./output/image/left/'+img_dir,left)
    cv2.imwrite('./output/image/right/'+img_dir,right)   
    cv2.imwrite('./output/image/BEV/'+img_dir,BEV_image)   

    if file_num % 10 == 0:
        cv2.imwrite('./output/global_map/'+img_dir,global_map)   

    cv2.namedWindow('BEV',cv2.WINDOW_NORMAL)
    # cv2.imshow('BEV',BEV_image)

    cv2.resizeWindow('BEV',400,400)

    cv2.imshow('global_map',global_map_draw)
    cv2.imshow('global_map_reduced',global_map_reduced_draw)
    # cv2.imshow('left',left)
    # cv2.imshow('right',right) 
    # cv2.imshow('disparity',disp)

    k = cv2.waitKey(100)

    if k == ord('s'):
        cv2.imwrite('global_map.png',global_map)
        print('global map saved!!')

    if k == 27:
        break
