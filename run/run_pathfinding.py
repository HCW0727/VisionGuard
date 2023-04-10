"""
D_star_Lite 2D
@author: huiming zhou
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np, cv2
import time

#setting atial value of previous_img
previous_img = [0,0]


from run.run_padding import run_PAD

class Env:
    def __init__(self,img_map):
        self.img = img_map
        self.y_range, self.x_range = self.img.shape
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        rows, cols = np.nonzero(self.img)
        obs = set(zip(cols, rows))

        return obs

class DStar:
    def __init__(self, img_map, s_start, s_goal, heuristic_type):
        self.img_map = img_map
        self.img_pad = run_PAD(img_map)
        

        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env(self.img_map)  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles
        self.x = self.Env.x_range
        self.y = self.Env.y_range

        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0

        for i in range( self.Env.x_range):
            for j in range( self.Env.y_range):
                self.rhs[(i, j)] = float("inf")
                self.g[(i, j)] = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)
        self.visited = set()
        self.count = 0
        # self.fig = plt.figure()

    

    def run(self):
        self.previous_img = self.img_map
        self.ComputePath()
        # self.plot_path(self.extract_path())
        return self.extract_path()
        

    def on_change(self,img_map):
        self.img_map = img_map
        self.img_pad = run_PAD(img_map)

        img_gap = self.img_map.astype(np.int16) - self.previous_img.astype(np.int16)

        # self.previous_img[148,4] = 0

        self.previous_img = self.img_map

        px_added = np.argwhere(img_gap > 0)
        px_removed = np.argwhere(img_gap < 0)

        for y,x in px_added:
            self.obs.add((x, y))
            self.g[(x, y)] = float("inf")
            self.rhs[(x, y)] = float("inf")
            for s in self.get_neighbor((x, y)):
                self.UpdateVertex(s)

        for y,x in px_removed:
            self.obs.remove((x, y))
            self.UpdateVertex((x, y))
            for s in self.get_neighbor((x, y)):
                self.UpdateVertex(s)
        

        s_curr = self.s_start
        s_last = self.s_start
        i = 0
        path = [self.s_start]

        self.km += self.h(s_last, s_curr)
        s_last = s_curr
        
        i += 1

        self.count += 1
        self.visited = set()
        self.ComputePath()

        visited = set()
        while s_curr != self.s_goal:
            visited.add(s_curr)
            s_list = {}
            for s in self.get_neighbor(s_curr):
                if s not in visited:
                    s_list[s] = self.g[s] + self.cost(s_curr, s)
            s_curr = min(s_list, key=s_list.get)
            path.append(s_curr)

        # self.plot_path(path)

        

        return path

    def ComputePath(self):
        st = time.time()
        count = 0
        max_iterations = 10000  # 최대 반복 횟수를 설정

        self.visited = set()

        while count < max_iterations:
            count += 1
            s, v = self.TopKey()


            if v >= self.CalculateKey(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            k_old = v
            self.U.pop(s)

            if k_old < self.CalculateKey(s):
                self.U[s] = self.CalculateKey(s)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)
            else:
                self.g[s] = float("inf")
                self.UpdateVertex(s)
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)

    def UpdateVertex(self, s):
        
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            for x in self.get_neighbor(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.cost(s, x))
        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.CalculateKey(s)

    def CalculateKey(self, s):
        return [min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal):
        heuristic_type = self.heuristic_type  # heuristic type
        if heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        elif heuristic_type == 'octile_distance':
            dx = abs(s_start[0] - s_goal[0])
            dy = abs(s_start[1] - s_goal[1])

            return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
        else:
            return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])


    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """
        # if self.is_collision(s_start, s_goal):
        #     # return float("inf")
        #     return 255


        bias = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
        s_movement = np.array(s_start) - np.array(s_goal)

        return self.img_pad[s_goal[1],s_goal[0]]/3 + bias

        if s_movement[0] == 0 or s_movement[1] == 0:
            return self.img_pad[s_goal[1],s_goal[0]]/3 + bias
        
        if s_start[0] != s_goal[0] and s_start[1] != s_goal[1]:
            if s_goal[0] - s_start[0] == s_start[1] - s_goal[1]:
                s1 = (min(s_start[0], s_goal[0]), min(s_start[1], s_goal[1]))
                s2 = (max(s_start[0], s_goal[0]), max(s_start[1], s_goal[1]))
                
            else:
                s1 = (min(s_start[0], s_goal[0]), max(s_start[1], s_goal[1]))
                s2 = (max(s_start[0], s_goal[0]), min(s_start[1], s_goal[1]))



            return min(math.hypot(self.img_pad[s1[1],s1[0]],self.img_pad[s_goal[1],s_goal[0]]),math.hypot(self.img_pad[s2[1],s2[0]],self.img_pad[s_goal[1],s_goal[0]]))/3 + bias
            # return self.img_pad[s_goal[1],s_goal[0]]/3 + bias
    

    def is_collision(self, s_start, s_end):
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def get_neighbor(self, s):
        nei_list = set()
        for u in self.u_set:
            s_next = tuple([s[i] + u[i] for i in range(2)])
            if s_next not in self.obs and s_next:
                if 0 <= s_next[0] < self.Env.x_range and 0 <= s_next[1] < self.Env.y_range:
                    nei_list.add(s_next)

        return nei_list

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(500):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_goal:
                break

        return list(path)

    
    def plot_path(self, path):
        self.img_map[self.img_map == 123] = 0
        
        for px, py in path:
            self.img_map[py][px] = 123


    
def main():
    # img_map = cv2.imread('pathfinding/saved_image.png',cv2.IMREAD_GRAYSCALE)

    
    img_map = resize_img(cv2.imread('output/global_map/000150.png',cv2.IMREAD_GRAYSCALE))

    height,width = img_map.shape
    
    s_start = (15, height-5)
    s_goal = (width-5, 15)

    dstar = DStar(img_map, s_start, s_goal,  "euclidean")
    dstar.run()
    
    print('change applied!')

def resize_img(img):
    h, w = img.shape[:2]
    nonzero_coords = np.nonzero(img)
    reduced_coords = (np.array(nonzero_coords) / 3).astype(np.uint8)
    img_reduced = np.zeros((h // 3, w // 3), dtype=np.uint8)
    img_reduced[reduced_coords[0], reduced_coords[1]] = 130

    return img_reduced


if __name__ == '__main__':
    main()
