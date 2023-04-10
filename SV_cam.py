import cv2
import time
import numpy as np

from run import run_CGI_STEREO,run_disparity2bev,run_global_map
from run import run_visual_odometry, run_pathfinding, run_size

webcam = cv2.VideoCapture(0)
middle_w = int(webcam.get(3)/2)

images_prev = None
path = None
current_angle = 0
one_px = 0.33
file_num = 0

start_point = (15, 85)
goal_point = (85, 15)

while webcam.isOpened():
    file_num += 1
    
    status,frame = webcam.read()
    left_image = frame[:,middle_w:]
    right_image = frame[:,:middle_w]

    ##################################################
    #Stereo to Disparity
    left,right,disp = run_CGI_STEREO.run_model(left_image,right_image)
    images_curr = [left,right,disp]
    
    ##################################################
    #Disparity to BEV
    BEV_image = run_disparity2bev.disp2bev(disp)

    ##################################################
    #BEV to global map(SVO, PathFinding)

    #Initial Image
    if images_prev == None:
        images_prev = images_curr
        continue

    #Visual Odometry : returns Moved x, moved z, rotated angle
    moved_x,moved_z,rotated_angle = run_visual_odometry.run_VO(images_prev,images_curr)
    images_prev = images_curr

    #Coordinate matching (for counterclockwise)
    rotated_angle *= -1
    
    #Convert a relative coordinate system to an absolute coordinate system
    theta_rad = np.deg2rad(-current_angle)
    rot_matrix = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                            [0, 1, 0],
                            [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
    
    input_vec = np.array([[moved_x], [0], [moved_z]])
    output_vec = np.dot(rot_matrix, input_vec)

    #Convert Meter to Pixel
    moved_x,moved_z = output_vec[0][0]/one_px, output_vec[2][0]/one_px

    #Since the moved value is the coordinate system of the previous image, rotated_angle is added after processing
    current_angle += rotated_angle
    moved_x,moved_z,current_angle = round(moved_x),round(moved_z),round(current_angle)

    #Get global map based on SVO information
    global_map = run_global_map.overlap(global_map,BEV_image,moved_x,moved_z,current_angle)

    #Downsampling image for PathFinding
    global_map_reduced = run_size.resize_img(global_map)
    ##################################################
    #BEV to global map(SVO, PathFinding)
    if path == None:
        dstar = run_pathfinding.DStar(global_map_reduced,start_point,goal_point,"octile_distance")
        path = dstar.run()

    else:
        path = dstar.on_change(global_map_reduced)

    ##################################################
    #Visualize global map
    global_map_draw = global_map.copy()
    for px,py in run_size.expand_path(path):
        global_map_draw[py,px] = 255

    #imwrite 
    file_num = str(file_num).zfill(6)
    cv2.imwrite('./output/webcam/disp/'+str(file_num).zfill(6),disp)
    cv2.imwrite('./output/webcam/left/'+str(file_num).zfill(6),left)
    cv2.imwrite('./output/webcam/right/'+str(file_num).zfill(6),right)   
    cv2.imwrite('./output/webcam/BEV/'+str(file_num).zfill(6),BEV_image)  

    if file_num % 10 == 0:
        cv2.imwrite('./output/global_map/'+str(file_num).zfill(6),global_map)


    #imshow
    cv2.imshow('global_map',global_map_draw)
    cv2.imshow('left',left)
    cv2.imshow('right',right) 
    cv2.imshow('disparity',disp)

    k = cv2.waitKey(100)
    if k == 27:
        break
