import cv2
from PIL import Image
from datasets.data_io import get_transform
from models import __models__
import torch
import torch.nn as nn
import os
import numpy as np

##################################################

# left_dir = 'Test_img/car_left.png'
# right_dir = 'Test_img/car_right.png'

# left_dir = 'Test_img/dt2_left.png'
# right_dir = 'Test_img/dt2_right.png'

# left_dir = 'Test_img/unpaved_L.png'
# right_dir = 'Test_img/unpaved_R.png'

# left_dir = '/Users/huhchaewon/python_projects/calibration/Sidewalk_img/calibrated/left6.png'
# right_dir = '/Users/huhchaewon/python_projects/calibration/Sidewalk_img/calibrated/right6.png'

left_dir = '/Users/huhchaewon/python_projects/calibration/5/left0.png'
right_dir = '/Users/huhchaewon/python_projects/calibration/5/right0.png'

# left_dir = '/Users/huhchaewon/python_projects/datasets/test/HR/outleft/0071.png'
# right_dir = '/Users/huhchaewon/python_projects/datasets/test/HR/outright/0071.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = __models__['CGI_Stereo'](192)
model = nn.DataParallel(model)
model.to(device)

ckpt_used = [fn for fn in os.listdir('./ckpt') if fn.endswith(".ckpt")]
ckpt_used = sorted(ckpt_used, key=lambda x: int(x.split('_')[-1].split('.')[0]))

state_dict = torch.load(os.path.join('./ckpt',ckpt_used[-1]),map_location=device)

model.load_state_dict(state_dict['model'])

model.eval()

##################################################
#preprocessing

left = Image.open(left_dir).convert('RGB')
right = Image.open(right_dir).convert('RGB')

original_width,original_height = left.size

print(original_width,original_height)
if original_width < original_height * 2:
    new_height = 256
    new_width = int(original_width * (new_height / original_height))
else:
    new_width = 512
    new_height = int(original_height * (new_width / original_width))

left = left.resize((new_width,new_height),resample=Image.Resampling.NEAREST).crop((0,0,512,256))
right = right.resize((new_width,new_height),resample=Image.Resampling.NEAREST).crop((0,0,512,256))

left.save('leftresult.png')
right.save('rightresult.png')

processed = get_transform()
left_img = processed(left)
right_img = processed(right)

##################################################

imgL = left_img.unsqueeze(0).to(device)
imgR = right_img.unsqueeze(0).to(device)


disp_ests = model(imgL,imgR)[-1].squeeze()
disp_est_np = disp_ests.detach().cpu()
disp_est_np = np.array(disp_est_np,dtype=np.float32)

print(disp_est_np)

disp_est_uint = np.round(disp_est_np *256 * 1).astype(np.uint16)

result = Image.fromarray(disp_est_uint)
result.save("result_disp.png", format="PNG")
