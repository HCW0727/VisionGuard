import numpy as np

def calculate_optimal_angle(user_pos, user_velocity, optimal_path):
    current_pos = np.array(user_pos)
    current_velocity = np.array(user_velocity)
    optimal_path = np.array(optimal_path)

    distance_to_path = np.linalg.norm(optimal_path - current_pos, axis=1)
    closest_point_index = np.argmin(distance_to_path)

    speed = np.linalg.norm(current_velocity)
    angle_range = (np.pi / 4) * (1 - min(1, speed / 10))

    path_direction = optimal_path[closest_point_index] - current_pos
    user_direction = current_velocity / speed

    optimal_angle = np.arccos(np.clip(np.dot(path_direction, user_direction), -1, 1))
    optimal_angle = np.clip(optimal_angle, -angle_range, angle_range)

    return optimal_angle

user_position = [1, 0]
user_velocity = [0, 1]
optimal_path = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]

optimal_angle = calculate_optimal_angle(user_position, user_velocity, optimal_path)

print("Optimal walking angle:", np.rad2deg(optimal_angle))
