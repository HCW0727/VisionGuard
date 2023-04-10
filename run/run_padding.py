import numpy as np
import time

def run_PAD(img):
    # img = cv2.imread('test_img.png', cv2.IMREAD_GRAYSCALE)

    height, width = img.shape[:2]

    mask = (img == 130)
    pad_range = 5
    padded_img = np.pad(img, pad_range, mode='constant', constant_values=0).astype(np.int32)
    padded_mask = np.pad(mask, pad_range, mode='constant', constant_values=0)

    for dy in range(-pad_range+1, pad_range):
        for dx in range(-pad_range+1, pad_range):
            if dx == 0 and dy == 0:
                continue
            # 가중치 공식 변경
            weighted_mask = (pad_range - max(np.abs(dx), np.abs(dy))) * 5
            current_region = padded_img[pad_range-1 + dy:height + pad_range-1 + dy, pad_range-1 + dx:width + pad_range-1 + dx]
            new_values = (padded_mask[pad_range-1:height + pad_range-1, pad_range-1:width + pad_range-1] * weighted_mask).astype(np.int32)
            padded_img[pad_range-1 + dy:height + pad_range-1 + dy, pad_range-1 + dx:width + pad_range-1 + dx] = np.maximum(current_region, new_values)

    # padding 제거 변경 (7로 변경)
    img = padded_img[pad_range-1:height + pad_range-1, pad_range-1:width + pad_range-1].clip(0, 255).astype(np.uint8)
    return img
 