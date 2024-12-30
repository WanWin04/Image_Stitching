# evaluation.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import model as m
import os

def calculate_psnr_ssim(image1, image2):
    """
    Calculates PSNR and SSIM between two images.

    Args:
        image1: The first image (numpy array).
        image2: The second image (numpy array).

    Returns:
        A tuple containing the PSNR and SSIM values.
    """

    # Ensure images are in uint8 format
    image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Calculate PSNR and SSIM
    psnr_value = psnr(image1, image2, data_range=255)
    ssim_value = ssim(image1, image2, win_size=3, data_range=255, multichannel=True)

    return psnr_value, ssim_value

def evaluate_feature_based(image1_path, image2_path):
    """
    Evaluates the feature-based image stitching method by calculating PSNR and SSIM.

    Args:
        image1_path: Path to the first input image.
        image2_path: Path to the second input image.
    """

    # Read input images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Convert to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 1. Run feature-based stitching to get homography
    image_stitching = m.ImageStitching(img1, img2)
    _, query_photo_gray = image_stitching.give_gray(img1)
    _, train_photo_gray = image_stitching.give_gray(img2)

    keypoints_train_image, features_train_image = image_stitching._sift_detector(
        train_photo_gray
    )
    keypoints_query_image, features_query_image = image_stitching._sift_detector(
        query_photo_gray
    )

    matches = image_stitching.create_and_match_keypoints(
        features_train_image, features_query_image
    )

    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )

    if M is None:
        print("Error: Cannot compute homography.")
        return

    (matches, homography_matrix, status) = M

    # 2. Create a mask for the overlapping region
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    warped_gray_img2 = cv2.warpPerspective(gray_img2, homography_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    mask = (warped_gray_img2 > 0).astype(np.uint8)

    # Resize mask to match img1's size BEFORE merging channels
    mask_resized = cv2.resize(mask, (img1.shape[1], img1.shape[0]))

    # Convert resized mask to 3 channels
    mask = cv2.merge([mask_resized, mask_resized, mask_resized])

    # 3. Apply mask to img1 and warp2
    masked_img1 = cv2.bitwise_and(img1, mask)
    warp2 = cv2.warpPerspective(img2, homography_matrix, (img1.shape[1], img1.shape[0]))
    masked_warp2 = cv2.bitwise_and(warp2, mask)

    # Ensure images are within the valid range [0, 255] and are of type uint8
    masked_img1 = np.clip(masked_img1, 0, 255).astype(np.uint8)
    masked_warp2 = np.clip(masked_warp2, 0, 255).astype(np.uint8)

    # 4. Calculate PSNR and SSIM
    psnr_val, ssim_val = calculate_psnr_ssim(masked_img1, masked_warp2)

    print(f"PSNR: {psnr_val:.2f}")
    print(f"SSIM: {ssim_val:.2f}")

if __name__ == "__main__":
    # Define the paths to the input images using os.path.join for cross-platform compatibility
    current_dir = os.path.dirname(__file__)
    image1_path = os.path.join(current_dir, "..","..","data", "UDIS-D", "testing", "input1", "000002.jpg")
    image2_path = os.path.join(current_dir, "..","..","data", "UDIS-D", "testing", "input2", "000002.jpg")

    # Check if the files exist
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("Error: One or both of the image files do not exist.")
        print(f"Image 1 path: {image1_path}")
        print(f"Image 2 path: {image2_path}")
    else:
        evaluate_feature_based(image1_path, image2_path)