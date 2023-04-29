import cv2
import numpy as np

def resize_img(img):
    h, w = img.shape[:2]
    nonzero_coords = np.nonzero(img)
    reduced_coords = (np.array(nonzero_coords) / 3).astype(np.uint8)
    img_reduced = np.zeros((h // 3, w // 3), dtype=np.uint8)
    img_reduced[reduced_coords[0], reduced_coords[1]] = 130

    return img_reduced

# def expand_path(path):
#     scale = 3
#     new_path = []
#     for x,y in path:
#         new_path.append([(x+1)*scale-1,(y+1)*scale-1])
#     return new_path

def expand_path(path, scale=3):
    new_path = []
    for i in range(len(path) - 1):
        current_point = np.array([(path[i][0]+1)*scale-1,(path[i][1]+1)*scale-1])
        next_point = np.array([(path[i+1][0]+1)*scale-1,(path[i+1][1]+1)*scale-1])

        delta_x, delta_y = next_point - current_point
        num_steps = max(abs(delta_x), abs(delta_y))

        interpolated_points = [np.round(current_point + (next_point - current_point) * t / num_steps).astype(int) for t in range(1, num_steps + 1)]

        new_path.append(current_point.tolist())
        new_path.extend(interpolated_points)

    new_path.append(path[-1])

    return new_path
