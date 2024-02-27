#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Triangulate_Points:
    def __init__(self, debug_op, pair) -> None:
        self.debug_op = debug_op
        self.pair = pair

    def plot_3d(self, points_3d):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point in points_3d:
            x, y, z = point
            ax.scatter(x, y, z, s=1)
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title(f"3D Points between Image Pair ({self.pair[0]}, {self.pair[1]})")

        plt.show()
    
    def common_points2d(self, prev2, curr1):
        matched_points = []
        matched_idx = []

        for i, j in enumerate(prev2):
            for x, y in enumerate(curr1):
                if np.array_equal(j, y):
                    matched_points.append(j)
                    matched_idx.append([i, x])
        
        return matched_points, matched_idx

    def common_points3d(self, pts_3d, indices):
        matched_points3d = []

        for i, _ in indices:
            matched_points3d.append(pts_3d[i].tolist())
        
        return matched_points3d

    def remove_outlier_3D_points(self,all_points, max_depth=20):
        mask = (all_points[:, 2] > 0) & np.all(all_points <= max_depth, axis=1)
        filtered_points = all_points[mask]

        return filtered_points, mask

    def triangulate(self, K, R1, R2, t1, t2, pts1, pts2):
        P1 = np.dot(K, np.hstack((R1, t1)))
        P2 = np.dot(K, np.hstack((R2, t2)))

        points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        return points_4d_homogeneous
    
    def find_camera_pose(self, pts_3d, pts_2d, K):
        _, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, None)
        R, _ = cv2.Rodrigues(rvec)

        return R, tvec