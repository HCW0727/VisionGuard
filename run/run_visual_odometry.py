from operator import imod
import os
import numpy as np
import cv2,math
# from data_structure import graph, node
from scipy.optimize import least_squares

# from lib.visualization import plotting
# from lib.visualization.video import play_trip

# from bundle_adjustment_solution import run_BA

from tqdm import tqdm
import time


class VisualOdometry():
    def __init__(self,images_1,images_2):
        #[left,right,disparity]
        self.image_l1 = images_1[0]
        self.image_l2 = images_2[0]

        self.image_r1 = images_1[1]
        self.image_r2 = images_2[1]

        self.image_d1 = np.divide(images_1[2],256)
        self.image_d2 = np.divide(images_2[2],256)

        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib('/Users/huhchaewon/Datasets/00/calib.txt')
        
        
        self.fastFeatures = cv2.FastFeatureDetector_create()

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        self.points3D = []
        self.points2D = []
        self.camIndex = 0
        self.keypoints_to_follow = [], []
        self.data = []


    @staticmethod 
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            P_l[0,2],P_l[1,2] = 512//2,256//2
            
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            P_r[0,2],P_r[1,2] = 512//2,256//2
            K_r = P_r[0:3, 0:3]

        return K_l, P_l, K_r, P_r


    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()

        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10] # ---------------------------------------------- only one keypoint is saved at the moment
            return keypoints
        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=6):
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
    
        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w, _ = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]
        
        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        ##################################################
        focal_x,focal_y,baseline = self.K_l[0,0],self.K_l[1,1],0.54
        disp1 = self.image_d1
        disp2 = self.image_d2

        height,width = disp1.shape
        center_y,center_x = height//2,width//2

        Q1,Q2 = [],[]
        for p_x,p_y in q1_l:
            z_point = (focal_x * baseline) /  disp1[int(p_y),int(p_x)]
            x_point = (p_x - center_x) * z_point / focal_x 
            y_point = (p_y - center_y) * z_point / focal_y
            Q1.append([x_point,y_point,z_point])

        for p_x,p_y in q2_l:
            z_point = (focal_x * baseline) /  disp2[int(p_y),int(p_x)]
            x_point = (p_x - center_x) * z_point / focal_x 
            y_point = (p_y - center_y) * z_point / focal_y
            Q2.append([x_point,y_point,z_point])

        Q1,Q2 = np.array(Q1),np.array(Q2)

        
        ##################################################

        # # Triangulate points from i-1'th image
        # Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # # Un-homogenize
        # Q1 = np.transpose(Q1[:3] / Q1[3])



        # # Triangulate points from i'th image
        # Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # # Un-homogenize
        # Q2 = np.transpose(Q2[:3] / Q2[3])

        # print(Q1)

        for i in range(len(Q2) - 1, 0, -1):
            if (Q2[i][2] > 75 or Q1[i][2] > 75):
                Q2 = np.delete(Q2, i, axis=0)
                q2_l = np.delete(q2_l,i, axis=0)
                q2_r = np.delete(q2_r,i, axis=0)
                Q1 = np.delete(Q1, i, axis=0)
                q1_l = np.delete(q1_l,i, axis=0)
                q1_r = np.delete(q1_r,i, axis=0)    



        return Q1, Q2, q1_l, q1_r, q2_l, q2_r


    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix



    def get_pose(self):
        img_l1,img_l2 = self.image_l1,self.image_l2
        img_r1,img_r2 = self.image_r1,self.image_r2
        img_d1,img_d2 = self.image_d1,self.image_d2

        
        # Get the tiled keypoints
        kp_l1 = self.get_tiled_keypoints(img_l1, 10, 10)

        # filter good keypoints
        tp1_l, tp2_l = self.track_keypoints(img_l1, img_l2, kp_l1)

        # print(len(kp_l1),len(tp1_l))
        
        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l,img_d1, img_d2)

        # Calculate the 3D points
        Q1, Q2, tp1_l, tp1_r, tp2_l, tp2_r = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        
        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
      
        return transformation_matrix

def run_VO(images_1,images_2):
    vo = VisualOdometry(images_1,images_2)

    # A = np.array([[1.000000e+00 ,9.043680e-12 ,2.326809e-11 ,5.551115e-17],
    # [9.043683e-12 ,1.000000e+00 ,2.392370e-10 ,3.330669e-16],
    # [2.326810e-11 ,2.392370e-10 ,9.999999e-01 ,-4.440892e-16],
    # [0, 0, 0, 1]])

    # B = np.array([[9.999978e-01 ,5.272628e-04 ,-2.066935e-03 ,-4.690294e-02],
    #               [-5.296506e-04 ,9.999992e-01 ,-1.154865e-03 ,-2.839928e-02],
    #                [2.066324e-03, 1.155958e-03 ,9.999971e-01 ,8.586941e-01],
    #                [0,0,0,1]])
    transf = vo.get_pose()

    moved_x,moved_z = transf[0,3],transf[2,3]
    r31,r32,r33 = transf[2,0],transf[2,1],transf[2,2]

    angle = math.atan2(-r31, math.sqrt(r32**2 + r33**2))
    angle_degrees = angle * 180 / math.pi

    return [moved_x,moved_z,angle_degrees]


# if __name__ == "__main__":
#     start_time = time.time()
#     main()
#     print('time : ',time.time()-start_time)
