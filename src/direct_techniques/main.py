import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def compute_gradients(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def feathered_cost_function(H, img1, img2):
    H = H.reshape(3, 3)
    warped_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    grad1_x, grad1_y = compute_gradients(img1)
    grad2_x, grad2_y = compute_gradients(warped_img2)

    cost = np.sum(np.abs(grad1_x - grad2_x)) + np.sum(np.abs(grad1_y - grad2_y))
    
    return cost

def optimize_homography(img1, img2, H_init):
    result = minimize(feathered_cost_function, H_init.flatten(), args=(img1, img2), method='L-BFGS-B')
    return result.x.reshape(3, 3)

def poisson_blend(img1, img2, mask):
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
    blended = cv2.seamlessClone(img2, img1, mask, center, cv2.NORMAL_CLONE)
    return blended

def blend_images(img1, img2, H):
    warped_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    mask = (warped_img2 > 0).astype(np.uint8) * 255

    blended = poisson_blend(img1, warped_img2, mask)
    
    return blended

def stitch_images(img1, img2):
    H_init = np.eye(3)

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
plt.title('Stitched Image using Gradient-domain Image Stitching')
plt.axis('off')
plt.show()

# https://link.springer.com/chapter/10.1007/978-3-540-24673-2_31?utm_source=getftr&utm_medium=getftr&utm_campaign=getftr_pilot&getft_integrator=sciencedirect_contenthosting
