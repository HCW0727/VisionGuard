import cv2,time
import numpy as np
from PIL import Image

pcd_filename = 'point_cloud.pcd'

img = cv2.imread('/Users/huhchaewon/python_projects/CGI-Stereo/result_disp.png',cv2.IMREAD_UNCHANGED)
img = img.astype(np.float32)/128.0


#preprocessing
# original_height,original_width = img.shape
# new_height = 256
# new_width = int(original_width * (new_height / original_height))

# img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
# img = img[:256, :512]


#Finding non-zero pixel / 0.07
nonzero_indices = np.nonzero(img)



#disparity point -> pcd
focal_x = 2.061940e+03
focal_y = 2.060674e+03


center_x = int(512/2)
center_y = int(256/2)

baseline = 0.54


#write pcd file
points = []

start_time = time.time()
for idx in range(len(nonzero_indices[0])):
    nonzero_x,nonzero_y = nonzero_indices[1][idx],nonzero_indices[0][idx]
    
    z_point = (focal_x * baseline) /  img[nonzero_y,nonzero_x]
    x_point = (nonzero_x - center_x) * z_point / focal_x
    y_point = (nonzero_y - center_y) * z_point / focal_y
    
    if 0 <= z_point <= 30 and -15 <= x_point <= 15 and 0.3 >= y_point >= -2:
        points.append([x_point,y_point,z_point])
        

print('time : ',time.time()-start_time)

#header
header = "# .PCD v.7 - Point Cloud Data file format\n"
header += "VERSION .7\n"
header += "FIELDS x y z\n"
header += "SIZE 4 4 4\n"
header += "TYPE F F F\n"
header += "COUNT 1 1 1\n"
header += "WIDTH {}\n".format(img.shape[1])
header += "HEIGHT {}\n".format(img.shape[0])
header += "VIEWPOINT 0 0 0 1 0 0 0\n"
header += "POINTS {}\n".format(img.shape[0] * img.shape[1])
header += "DATA ascii\n"

#write pcd file
points = np.array(points).astype(np.float32)
pcd_file = open(pcd_filename, 'w')
pcd_file.write(header)
for point in points:
    pcd_file.write("{} {} {}\n".format(point[0], point[1], point[2]))
pcd_file.close()


