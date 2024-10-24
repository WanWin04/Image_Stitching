import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def compute_initial_homography(img1, img2):
    h, w = img1.shape[:2]
    points1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
    points2 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
    
    H, status = cv2.findHomography(points1, points2)
    return H

def warp_image(img, H, shape):
    return cv2.warpPerspective(img, H, shape)

def objective_function(H, img1, img2):
    H = H.reshape(3, 3)
    warped_img2 = warp_image(img2, H, (img1.shape[1], img1.shape[0]))
    
    return np.sum((img1 - warped_img2) ** 2)

def optimize_homography(img1, img2, H_init):
    result = minimize(objective_function, H_init.flatten(), args=(img1, img2), method='L-BFGS-B')
    return result.x.reshape(3, 3)

def blend_images(img1, img2, H):
    warped_img2 = warp_image(img2, H, (img1.shape[1], img1.shape[0]))
    
    mask = (warped_img2 > 0).astype(np.uint8) * 255
    blended = cv2.seamlessClone(warped_img2, img1, mask, (img1.shape[1]//2, img1.shape[0]//2), cv2.NORMAL_CLONE)
    
    return blended

def stitch_images(img1, img2):
    H_init = compute_initial_homography(img1, img2)
    
    H_optimized = optimize_homography(img1, img2, H_init)
    
    blended_image = blend_images(img1, img2, H_optimized)
    
    return blended_image, H_optimized

img1 = cv2.imread('../../image/test1.jpg')
img2 = cv2.imread('../../image/test2.jpg')

img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

result, homography = stitch_images(img1, img2)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Stitched Image')
plt.axis('off')
plt.show()
