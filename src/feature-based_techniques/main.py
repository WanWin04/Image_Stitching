import cv2
import numpy as np
import matplotlib.pyplot as plt

class Features:
    def __init__(self, method, crossCheck=True):
        self.method = method
        self.crossCheck = crossCheck

    def detectAndDescribe(self, image):
        # Detect and extract features from the image
        if self.method == 'SIFT':
            descriptor = cv2.SIFT_create()
        elif self.method == 'ORB':
            descriptor = cv2.ORB_create()
        elif self.method == 'BRISK':
            descriptor = cv2.BRISK_create()
        elif self.method == 'AKAZE':
            descriptor = cv2.AKAZE_create()
        
        # get keypoints and descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)
        
        return (kps, features)
    
    def showKeypoints(self, img1, img2, kpsA, kpsB):
        # display the keypoints and features detected on both images
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
        ax1.imshow(cv2.drawKeypoints(img1,kpsA,None,color=(0,255,0)))
        ax1.set_xlabel("(a)", fontsize=14)
        ax2.imshow(cv2.drawKeypoints(img2,kpsB,None,color=(0,255,0)))
        ax2.set_xlabel("(b)", fontsize=14)
        plt.savefig('../output/keypoints.png')
        plt.close()

    def createMatcher(self):
        # Create and return a Matcher Object
        if self.method == 'SIFT':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=self.crossCheck)
        elif self.method == 'ORB' or self.method == 'BRISK' or self.method == 'AKAZE':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.crossCheck)
        return bf
    
    def matchKeyPointsBF(self, featuresA, featuresB):
        bf = self.createMatcher()
        best_matches = bf.match(featuresA,featuresB)
        # The points with small distance (more similarity) are ordered first in the vector
        rawMatches = sorted(best_matches, key = lambda x:x.distance)
        print("Raw matches (Brute force):", len(rawMatches))
        return rawMatches
    
    def matchKeyPointsKNN(self, featuresA, featuresB, ratio):
        bf = self.createMatcher()
        rawMatches = bf.knnMatch(featuresA, featuresB, 2)
        print("Raw matches (knn):", len(rawMatches))
        matches = []
        for m,n in rawMatches:
            # ensure the distance is within a certain ratio of each
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches
    
    def matchKeyPointsFlann(self, featuresA, featuresB, ratio):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        rawMatches = flann.knnMatch(featuresA,featuresB,k=2)
        print("Raw matches (flann):", len(rawMatches))
        matches = []
        for m,n in rawMatches:
            # ensure the distance is within a certain ratio of each
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches
    
    def showFeatureMatch(self, img1, img2, kpsA, kpsB, featuresA, featuresB, match, ratio=1):
        if match == 'BF':
            matches = self.matchKeyPointsBF(featuresA, featuresB)
            img3 = cv2.drawMatches(img1, kpsA, img2, kpsB, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif match == 'KNN':
            matches = self.matchKeyPointsKNN(featuresA, featuresB, ratio)
            img3 = cv2.drawMatches(img1, kpsA, img2, kpsB, np.random.choice(matches, 100), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif match == 'FLANN':
            matches = self.matchKeyPointsFlann(featuresA, featuresB, ratio)
            img3 = cv2.drawMatches(img1, kpsA, img2, kpsB, np.random.choice(matches, 100), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imwrite('../output/matches.jpg', img3)
        return matches
    
class Homography:
    def __init__(self, method='RANSAC', reprojThresh=5.0):
        self.method = method
        self.reprojThresh = reprojThresh

    def getHomography(self, kpsA, kpsB, matches):
        # Convert the keypoints to numpy arrays
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Compute the homography between the two sets of points
        (M, mask) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)
        return (M, mask)

class Blender:
    def __init__(self):
        pass

    def alpha_blend(self, img1, img2, mask):
        # Create a linearly increasing mask
        mask = np.zeros((img2.shape[0], img2.shape[1]), dtype=np.float32)
        mask[:, :int(img2.shape[1]/2)] = np.linspace(0, 1, int(img2.shape[1]/2))
        mask = mask.astype(np.float32)
        mask = 0.5
        result = cv2.addWeighted(img1, float(1-mask), img2, float(mask), 0)
        return result
    
    def gaussian_blend(self, img1, img2):
       # Compute the Gaussian pyramid for each image
        G1 = img1.copy()
        G2 = img2.copy()
        gp1 = [G1]
        gp2 = [G2]
        for i in range(6):
            G1 = cv2.pyrDown(G1)
            G2 = cv2.pyrDown(G2)
            gp1.append(G1)
            gp2.append(G2)

        # Compute the Laplacian pyramid for each image
        lp1 = [gp1[5]]
        lp2 = [gp2[5]]
        for i in range(5, 0, -1):
            size = (gp1[i - 1].shape[1], gp1[i - 1].shape[0])
            L1 = cv2.subtract(gp1[i - 1], cv2.pyrUp(gp1[i], dstsize=size))
            L2 = cv2.subtract(gp2[i - 1], cv2.pyrUp(gp2[i], dstsize=size))
            lp1.append(L1)
            lp2.append(L2)

        # Combine the left and right halves of each level of the Laplacian pyramid
        LS = []
        for l1, l2 in zip(lp1, lp2):
            rows, cols, dpt = l1.shape
            ls = np.hstack((l1[:, :cols//2], l2[:, cols//2:]))
            LS.append(ls)

        # Reconstruct the blended image from the Laplacian pyramid
        ls_ = LS[0]
        for i in range(1, 6):
            size = (LS[i].shape[1], LS[i].shape[0])
            ls_ = cv2.add(cv2.pyrUp(ls_, dstsize=size), LS[i])

        result = ls_

        return result 
    
    def seamless_cloning(self, img1, img2):
        # Create a mask for the center of the image
        mask = np.zeros(img1.shape, dtype=np.uint8)
        mask[img1.shape[0]//2-50:img1.shape[0]//2+50, img1.shape[1]//2-50:img1.shape[1]//2+50] = 255

        # Create a rough mask around the center of the image
        rough_mask = np.zeros(img1.shape, dtype=np.uint8)
        rough_mask[img1.shape[0]//2-100:img1.shape[0]//2+100, img1.shape[1]//2-100:img1.shape[1]//2+100] = 255

        # Use the rough mask to find the center of the image
        M = cv2.moments(rough_mask[:,:,0])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Use the center to create a seamless cloning mask
        result = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)

        return result
    
    def multiband_blend(self, img1, img2):
        # Create a rough mask around the center of the image
        rough_mask = np.zeros(img1.shape, dtype=np.uint8)
        rough_mask[img1.shape[0]//2-100:img1.shape[0]//2+100, img1.shape[1]//2-100:img1.shape[1]//2+100] = 255

        # Convert the rough_mask to the required type (8-bit unsigned integer)
        rough_mask = rough_mask.astype(np.uint8)

        # Use the rough mask to find the center of the image
        M = cv2.moments(rough_mask[:,:,0])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Create a multiband blender
        blender = cv2.detail_MultiBandBlender()
        print(rough_mask.shape, center)
        # Blend the images with the Assertion failed) mask.type() == CV_8U in function 'feed'
        blender.feed(img1, rough_mask, center)
        blender.feed(img2, rough_mask, center)
        result = blender.blend(rough_mask, center)

        return result
    

