import cv2
from PIL import Image
from datasets.data_io import get_transform
from models import __models__
import torch
import torch.nn as nn
import os
import numpy as np
import time


####################################################################################

l_mtx = np.array([[1.36688831e+03 ,0.00000000e+00 ,5.42896031e+02],
 [0.00000000e+00 ,1.37042747e+03 ,3.87374004e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

l_newCameramtx = np.array([[1.16126453e+03 ,0.00000000e+00 ,5.20710207e+02],
 [ 0.00000000e+00 ,1.13378918e+03 ,3.87108333e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

l_dist = np.array([[-4.50195983e-01  ,2.29675267e-01 ,-1.34943624e-03  ,9.91683482e-05, 8.18452763e-03]])

####################################################################################

r_mtx = np.array([[1.36922572e+03 ,0.00000000e+00 ,6.76968838e+02],
 [0.00000000e+00 ,1.36635145e+03 ,3.34266442e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

r_newCameramtx = np.array([[1.17232092e+03 ,0.00000000e+00 ,6.85258002e+02],
 [0.00000000e+00 ,1.15809595e+03 ,3.33400673e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

r_dist = np.array([[-4.34075416e-01  ,2.04086358e-01 ,1.70629664e-03 ,-3.54754716e-04 ,-1.60347753e-02]])


def run_model(left_img,right_img):
    start_time = time.time()

    ####################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = __models__['CGI_Stereo'](192)
    model = nn.DataParallel(model)
    model.to(device)
                
    ckpt_used = [fn for fn in os.listdir('./ckpt') if fn.endswith(".ckpt")]
    ckpt_used = sorted(ckpt_used, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    state_dict = torch.load(os.path.join('./ckpt',ckpt_used[-1]),map_location=device)

    model.load_state_dict(state_dict['model'])

    model.eval()
    ####################################################################################
 
    left = Image.fromarray(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    right = Image.fromarray(cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB))

    original_width,original_height = left.size

    if original_width < original_height * 2:
        new_width = 512
        new_height = int(original_height * (new_width / original_width))

        res_height = new_height - 256

        left = left.resize((new_width,new_height),resample=Image.Resampling.NEAREST).crop((0,res_height//2,512,new_height-res_height//2))
        right = right.resize((new_width,new_height),resample=Image.Resampling.NEAREST).crop((0,res_height//2,512,new_height-res_height//2))
    else:
        new_height = 256
        new_width = int(original_width * (new_height / original_height))

        res_width = new_width - 512

        left = left.resize((new_width,new_height),resample=Image.Resampling.NEAREST).crop((res_width//2,0,new_width-res_width//2,256))
        right = right.resize((new_width,new_height),resample=Image.Resampling.NEAREST).crop((res_width//2,0,new_width-res_width//2,256))

    

    processed = get_transform()
    left_img = processed(left)
    right_img = processed(right)

    ##################################################

    imgL = left_img.unsqueeze(0).to(device)
    imgR = right_img.unsqueeze(0).to(device)

    disp_ests = model(imgL,imgR)[-1].squeeze()
    disp_est_np = disp_ests.detach().cpu()
    disp_est_np = np.array(disp_est_np,dtype=np.float32)

    disp_est_uint = np.round(disp_est_np * 256 ).astype(np.uint16)

    # print('Stereo to Disaprity : ',time.time()-start_time)
    left_img_cv2 = cv2.cvtColor(np.array(left),cv2.COLOR_RGB2BGR)
    right_img_cv2 = cv2.cvtColor(np.array(right),cv2.COLOR_RGB2BGR)

    return left_img_cv2,right_img_cv2,disp_est_uint


        