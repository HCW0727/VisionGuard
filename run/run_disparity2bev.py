import cv2,time
import numpy as np
from PIL import Image

#disparity point -> pcd
focal_x = focal_y = 7.188560000000e+02

center_x = int(512/2)
center_y = int(256/2)

baseline = 0.54

def disp2bev(img):
    start_time = time.time()

    img = img.astype(np.float32)/256.0
    nonzero_indices = np.nonzero(img)

    img_bev = np.array([[0] * 100 for _ in range(100)],dtype=np.uint8)

    # img_bev[0,11],img_bev[0,88] = 255,255

    for idx in range(len(nonzero_indices[0])):
        nonzero_x,nonzero_y = nonzero_indices[1][idx],nonzero_indices[0][idx]
        
        z_point = (focal_x * baseline) /  img[nonzero_y,nonzero_x]
        x_point = (nonzero_x - center_x) * z_point / focal_x
        y_point = (nonzero_y - center_y) * z_point / focal_y

        

        
    
        if 0 <= z_point <= 30 and -15 <= x_point <= 15 and 0.5 >= y_point >= -2:
            x_point = int((float(x_point)+15)/30*100)
            z_point = 100-int(float(z_point)/30*100)

            if 0<=x_point<100 and 0<= z_point < 100:
                for cx in range(x_point,x_point+2):
                    for cz in range(z_point,z_point+2):
                        img_bev[cz,cx] = 130
                # img_bev[z_point,x_point] = 130

    # img_bev[-1,49:50] = 255

    # print('Disparity to BEV : ',time.time()-start_time)
    return img_bev