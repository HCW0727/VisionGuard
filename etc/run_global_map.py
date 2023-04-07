import cv2,time
import numpy as np

current_x = 50
current_z = 0
current_angle = 0

def get_rotated_mask(img):
    height,width = img.shape
    mask = (img==123)
    mask_img = np.zeros((width,height), dtype=np.uint8)
    mask_img[mask] = 255
    mask_img_inv = np.bitwise_not(mask_img)

    return mask_img,mask_img_inv

def get_roi_triangle():
    mask = np.zeros((100, 100), dtype=np.uint8)

    start_time = time.time()

    pts = np.array([[11, 0], [50, 99], [49, 99], [88, 0]], dtype=np.int32)

    cv2.fillPoly(mask, [pts], color=255)
    mask_inv = np.bitwise_not(mask)
    return mask,mask_inv

def rotate_img(img,angle):
    global M
    height, width = img.shape
    M = cv2.getRotationMatrix2D((width/2,height/2),angle,1)

    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    M[0,2] += (new_width / 2) - (width / 2)
    M[1,2] += (new_height / 2) - (height / 2)

    rotated_img = cv2.warpAffine(img, M, (new_width, new_height),borderValue = (123,123))

    return rotated_img,M

def overlap(org_img,overlap_img,moved_x,moved_z,moved_angle=0):
    global current_x,current_z,current_angle
    current_x += moved_x
    current_z += moved_z
    current_angle += moved_angle

    print('current_angle : ',current_angle)

    org_height,org_width = org_img.shape
    pre_height,pre_width = overlap_img.shape

    left_pad,right_pad,bottom_pad,top_pad = 0,0,0,0

    if moved_angle != 0:
        overlap_img, rotation_matrix = rotate_img(overlap_img,moved_angle)
        
        org_point = rotation_matrix @ np.array([pre_width//2,pre_height,1])
        print(org_point)

        pre_height,pre_width = overlap_img.shape
        new_point = [pre_width//2,pre_height]
        gap_matrix = np.array(new_point - org_point).astype(np.int8)

        changed_x = current_x + gap_matrix[0]
        changed_z = current_z - gap_matrix[1]
        
        changed_x -= pre_width//2

        # pre_height,pre_width = overlap_img.shape
        

    else:
        changed_x = current_x - pre_width//2
        changed_z = current_z


    if changed_x < 0:
        left_pad = abs(changed_x)
        changed_x = 0
        org_width += left_pad
    if changed_x + pre_width> org_width:
        right_pad = changed_x + pre_width - org_width
        org_width += right_pad
    if changed_z < 0:
        bottom_pad = abs(changed_z)
        changed_z = 0 
        org_height += bottom_pad
    if changed_z + pre_height > org_height:
        top_pad = changed_z + pre_height - org_height
        org_height += top_pad

    img_with_border = cv2.copyMakeBorder(org_img, top_pad, bottom_pad,left_pad , right_pad, cv2.BORDER_CONSTANT, value=255)


    # print(changed_z)
    org_height,org_width = img_with_border.shape
    changed_z = org_height - changed_z

    if moved_angle != 0:
        rotate_mask,rotate_mask_inv = get_rotated_mask(overlap_img)

        overlap_img_no_border = cv2.bitwise_and(rotate_mask_inv,overlap_img)

        masked_original = cv2.bitwise_and(img_with_border[changed_z-pre_height:changed_z,
                                    changed_x:changed_x+pre_width],rotate_mask)
        
        overlap_img = cv2.bitwise_or(overlap_img_no_border,masked_original)  

        cv2.imshow('test',img_with_border[changed_z-pre_height:changed_z,
                                    changed_x:changed_x+pre_width])
        cv2.imshow('test2',rotate_mask)
        cv2.imshow('test3',masked_original)
        cv2.imshow('test4',overlap_img_no_border) 
        
        cv2.waitKey(0)

    mask_triangle,mask_triangle_inv = get_roi_triangle()
    test_img = cv2.bitwise_and(mask_triangle,overlap_img)

    img_with_border[changed_z-pre_height:changed_z,
                     changed_x:changed_x+pre_width] = overlap_img
    

    print(np.max(img_with_border))
    
    return img_with_border


if __name__ == '__main__':
    start_time = time.time()
    org_img = cv2.imread('./output/image/BEV/000000.png', cv2.IMREAD_GRAYSCALE)

    # org_img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
    overlap_img = cv2.imread('./output/image/BEV/000001.png',cv2.IMREAD_GRAYSCALE)

    result = overlap(org_img,overlap_img,20,20,20)

    print('change applied!')
    # print(time.time() - start_time)