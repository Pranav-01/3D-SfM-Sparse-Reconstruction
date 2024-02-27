#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Finding_F_and_E:
    def __init__(self, debug_op) -> None:
        self.sift_detector = cv2.SIFT_create()
        self.debug_op = debug_op
    
    def use_sift(self, image):
        kp, des = self.sift_detector.detectAndCompute(image, None)
        return kp, des

    def knn_matching(self, pair, kp0, des0, kp1, des1, image1, image2):
        self.good_matches = []

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key=lambda x: x[0].distance)
        
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                self.good_matches.append(m)
        
        if self.debug_op:
            MatchImage = cv2.drawMatches(image1, kp0, image2, kp1, self.good_matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,)
            plt.imshow(MatchImage)
            plt.title(f"Matches between Image {pair[0] + 1} and Image {pair[1] + 1}")
            plt.show()

    def compute_E(self, kp0, kp1, pair, K):
        SRC = np.float32([kp0[m.queryIdx].pt for m in self.good_matches]).squeeze()
        DST = np.float32([kp1[m.trainIdx].pt for m in self.good_matches]).squeeze()

        E, mask = cv2.findEssentialMat(SRC, DST, K, cv2.RANSAC, 0.999, 1.0)
        _, R, t, _ = cv2.recoverPose(E, SRC, DST,K)
        inliers = SRC[mask.ravel() == 1]
        
        if self.debug_op:
            print(f"For Pair {pair}:-\nESSENTIAL MATRIX:\n{E}\nR_MATRIX:\n{R}\nt_MATRIX:\n{t}\nN_INLIERS: {len(inliers)}")

        return E, R, t, mask, SRC, DST, self.good_matches

    def enforce_rank2_matrix(self, matrix):
        U, D, Vt = np.linalg.svd(matrix)
        D[2] = 0
        return np.dot(U, np.dot(np.diag(D), Vt))