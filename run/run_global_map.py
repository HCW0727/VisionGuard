import cv2,time
import numpy as np

# current_x = 50
# current_z = 0

##################################################
# current_x = 51
# current_z = 318
# current_angle = 1
##################################################
global_map = cv2.imread('pathfinding/global_map.png',cv2.IMREAD_GRAYSCALE)
# current_angle = 0

def get_rotated_mask(img):
    height,width = img.shape
    mask = (img==123)
    mask_img = np.zeros((width,height), dtype=np.uint8)
    mask_img[mask] = 255
    mask_img_inv = np.bitwise_not(mask_img)

    return mask_img,mask_img_inv

#get pixels in triangle
def get_final_mask(img):
    _, mask = cv2.threshold(img, 1, 130, cv2.THRESH_BINARY)
    return mask

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

def overlap(org_img,overlap_img,current_x,current_z,current_angle):
    global global_map #,current_angle

    # current_x += moved_x
    # current_z += moved_z

    # print(round(current_x,2),round(current_z,2),round(current_angle,2))
    # current_angle += moved_angle

    # current_x,current_z,current_angle = round(current_x),round(current_z),round(current_angle)

    triangle_roi_mask,triangle_roi_mask_inv= get_roi_triangle()

    triangle_rotate_mask = np.array(triangle_roi_mask_inv / 255 * 123,dtype=np.uint8)
    
    overlap_img_no_triangle = cv2.bitwise_and(triangle_roi_mask,overlap_img)
    overlap_img = cv2.bitwise_or(triangle_rotate_mask,overlap_img_no_triangle)

    org_height,org_width = org_img.shape
    pre_height,pre_width = overlap_img.shape

    left_pad,right_pad,bottom_pad,top_pad = 0,0,0,0

    if current_angle != 0:
        overlap_img, rotation_matrix = rotate_img(overlap_img,current_angle)
        
        org_point = rotation_matrix @ np.array([pre_width//2,pre_height,1])

        pre_height,pre_width = overlap_img.shape 
        new_point = [pre_width//2,pre_height]
        
        gap_matrix = np.array(new_point - org_point).astype(np.int8)


        #current_x, current_z 일정 -> gap_matrix의 문제
        changed_x = current_x + gap_matrix[0]
        changed_z = current_z - gap_matrix[1]

        changed_x -= pre_width//2


    else:
        changed_x = current_x - pre_width//2
        changed_z = current_z


    if changed_x < 0:
        left_pad = abs(changed_x)
        changed_x = 0
        org_width += left_pad

        current_x += left_pad
    if changed_x + pre_width> org_width:
        right_pad = changed_x + pre_width - org_width
        org_width += right_pad

        # current_x -= right_pad
    if changed_z < 0:
        bottom_pad = abs(changed_z)
        changed_z = 0 
        org_height += bottom_pad

        current_z += bottom_pad
    if changed_z + pre_height > org_height:
        top_pad = changed_z + pre_height - org_height
        org_height += top_pad

        # current_z -= top_pad

    img_with_border = cv2.copyMakeBorder(org_img, top_pad, bottom_pad,left_pad , right_pad, cv2.BORDER_CONSTANT, value=0)

    org_height,org_width = img_with_border.shape
    changed_z = org_height - changed_z

    if current_angle != 0:
        rotate_mask,rotate_mask_inv = get_rotated_mask(overlap_img)

        overlap_img_no_border = cv2.bitwise_and(rotate_mask_inv,overlap_img)

        masked_original = cv2.bitwise_and(img_with_border[changed_z-pre_height:changed_z,
                                    changed_x:changed_x+pre_width],rotate_mask)
        
        overlap_img = cv2.bitwise_or(overlap_img_no_border,masked_original)  

    else:
        overlap_img_final = get_final_mask(overlap_img_no_triangle)
        
        masked_original = cv2.bitwise_and(img_with_border[changed_z-pre_height:changed_z,
                                        changed_x:changed_x+pre_width],triangle_roi_mask_inv)
        
        # overlap_img = cv2.bitwise_or(overlap_img_final,img_with_border[changed_z-pre_height:changed_z,
        #                                 changed_x:changed_x+pre_width])
        overlap_img = cv2.bitwise_or(overlap_img_final,masked_original)
    

    img_with_border[changed_z-pre_height:changed_z,
                     changed_x:changed_x+pre_width] = overlap_img
    
    img_with_border[(img_with_border != 130) & (img_with_border != 255)] =  0

    #global_map의 border 살리기
    _, res_mask = cv2.threshold(img_with_border, 100, 255, cv2.THRESH_BINARY)
    _, global_map_mask = cv2.threshold(global_map, 100, 255, cv2.THRESH_BINARY)

    res_img = cv2.bitwise_or(res_mask,global_map_mask)
    res_img[res_img==255] = 130

    return res_img


if __name__ == '__main__':
    start_time = time.time()
    org_img = cv2.imread('pathfinding/global_map.png', cv2.IMREAD_GRAYSCALE)

    # org_img = cv2.imread('./saved_image.png',cv2.IMREAD_GRAYSCALE)
    overlap_img = cv2.imread('./output/image/BEV/000001.png',cv2.IMREAD_GRAYSCALE)

    result = overlap(org_img,overlap_img,0,0,0)

    
    cv2.imshow('result',result)
    cv2.waitKey(0)

    print('change applied!')
    # print(time.time() - start_time)