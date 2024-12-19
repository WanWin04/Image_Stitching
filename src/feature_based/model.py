import cv2
import numpy as np
import sys

class ImageStitching:
    """containts the utilities required to stitch images"""

    def __init__(self, query_photo, train_photo):
        super().__init__()
        width_query_photo = query_photo.shape[1]
        width_train_photo = train_photo.shape[1]
        lowest_width = min(width_query_photo, width_train_photo)
        smoothing_window_percent = 0.10 # consider increasing or decreasing[0.00, 1.00] 
        self.smoothing_window_size = max(100, min(smoothing_window_percent * lowest_width, 1000))

    def give_gray(self, image):
        """receives an image array and returns grayscaled image

        Args:
            image (numpy array): array of images

        Returns:
            image (numpy array): same as image input
            photo_gray (numpy array): grayscaled images
        """
        photo_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image, photo_gray



    @staticmethod
    def _sift_detector(image):
        """Applies SIFT algorithm to the given image

        Args:
            image (numpy array): input image

        Returns:
            keypoints, features
        """
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(image, None)

        return keypoints, features

    def create_and_match_keypoints(self, features_train_image, features_query_image):
        """Creates and Matches keypoints from the SIFT features using Brute Force matching
        by checking the L2 norm of the feature vector

        Args:
            features_train_image: SIFT features of train image
            features_query_image: SIFT features of query image

        Returns:
            matches (List): matches in features of train and query image
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        best_matches = bf.match(features_train_image, features_query_image)
        raw_matches = sorted(best_matches, key=lambda x: x.distance)

        return raw_matches

    def compute_homography(
        self, keypoints_train_image, keypoints_query_image, matches, reprojThresh
    ):
        """Computes the Homography to map images to a single plane,
        uses RANSAC algorithm to find the best matches iteratively.

        Args:
            keypoints_train_image: keypoints found using SIFT in train image
            keypoints_query_image: keypoints found using SIFT in query image
            matches: matches found using Brute Force
            reprojThresh: threshold for error

        Returns:
            M (Tuple): (matches, Homography matrix, status)
        """
        keypoints_train_image = np.float32(
            [keypoint.pt for keypoint in keypoints_train_image]
        )
        keypoints_query_image = np.float32(
            [keypoint.pt for keypoint in keypoints_query_image]
        )

        if len(matches) >= 4:
            points_train = np.float32(
                [keypoints_train_image[m.queryIdx] for m in matches]
            )
            points_query = np.float32(
                [keypoints_query_image[m.trainIdx] for m in matches]
            )

            H, status = cv2.findHomography(
                points_train, points_query, cv2.RANSAC, reprojThresh
            )

            return (matches, H, status)

        else:
            print(f"Minimum match count not satisfied cannot get homopgrahy")
            return None

    def create_mask(self, query_image, train_image, version):
        """Creates the mask using query and train images for blending the images,
        using a gaussian smoothing window/kernel

        Args:
            query_image (numpy array)
            train_image (numpy array)
            version (str) == 'left_image' or 'right_image'

        Returns:
            masks
        """
        height_query_photo = query_image.shape[0]
        width_query_photo = query_image.shape[1]
        width_train_photo = train_image.shape[1]
        height_panorama = height_query_photo
        width_panorama = width_query_photo + width_train_photo
        offset = int(self.smoothing_window_size / 2)
        barrier = query_image.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version == "left_image":
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height_panorama, 1)
            )
            mask[:, barrier + offset :] = 1
        return cv2.merge([mask, mask, mask])

    def blending_smoothing(self, query_image, train_image, homography_matrix):
        """blends both query and train image via the homography matrix,
        and ensures proper blending and smoothing using masks created in create_masks()
        to give a seamless panorama.

        Args:
            query_image (numpy array)
            train_image (numpy array)
            homography_matrix (numpy array): Homography to map images to a single plane

        Returns:
            panoramic image (numpy array)
        """
        height_img1 = query_image.shape[0]
        width_img1 = query_image.shape[1]
        width_img2 = train_image.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(query_image, train_image, version="left_image")
        panorama1[0 : query_image.shape[0], 0 : query_image.shape[1], :] = query_image
        panorama1 *= mask1
        mask2 = self.create_mask(query_image, train_image, version="right_image")
        panorama2 = (
            cv2.warpPerspective(
                train_image, homography_matrix, (width_panorama, height_panorama)
            )
            * mask2
        )
        result = panorama1 + panorama2

        # remove extra blackspace
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        final_result = result[min_row:max_row, min_col:max_col, :]

        return final_result

def read(image_dir_list):
    """reads the images dir list and returns the images array as List

    Args:
        image_dir_list (List): image list read through cmdline

    Returns:
        images_list(List): List of numpy array of Images
        len(images_list) (int): no of images read through cmdline
    """
    images_list = []
    for image_dir in image_dir_list:
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_list.append(image)

    return images_list, len(images_list)

def recurse(image_list, no_of_images):
    """Recursive function to get panorama of multiple images

    Args:
        image_list (List): list of numpy array of images
        no_of_images (int): no of images read through cmdline

    Returns:
        result (numpy array): RGB panoramic image
    """
    if no_of_images == 2:
        result, mapped_image = forward(
            query_photo=image_list[no_of_images - 2],
            train_photo=image_list[no_of_images - 1],
        )

        return result, mapped_image
    else:
        result,_ = forward(
            query_photo=image_list[no_of_images - 2],
            train_photo=image_list[no_of_images - 1],
        )
        result_int8 = np.uint8(result)
        result_rgb = cv2.cvtColor(result_int8, cv2.COLOR_BGR2RGB)
        image_list[no_of_images - 2] = result_rgb

        return recurse(image_list, no_of_images - 1)
    
def forward(query_photo, train_photo):
    """Runs a forward pass using the ImageStitching() class in utils.py.
    Takes in a query image and train image and runs entire pipeline to return
    a panoramic image.

    Args:
        query_photo (numpy array): query image
        train_photo (nnumpy array): train image

    Returns:
        result image (numpy array): RGB result image
    """
    image_stitching = ImageStitching(query_photo, train_photo)
    _, query_photo_gray = image_stitching.give_gray(query_photo)  # left image
    _, train_photo_gray = image_stitching.give_gray(train_photo)  # right image

    keypoints_train_image, features_train_image = image_stitching._sift_detector(
        train_photo_gray
    )
    keypoints_query_image, features_query_image = image_stitching._sift_detector(
        query_photo_gray
    )

    matches = image_stitching.create_and_match_keypoints(
        features_train_image, features_query_image
    )

    mapped_feature_image = cv2.drawMatches(
                        train_photo,
                        keypoints_train_image,
                        query_photo,
                        keypoints_query_image,
                        matches[:100],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    M = image_stitching.compute_homography(
        keypoints_train_image, keypoints_query_image, matches, reprojThresh=4
    )

    if M is None:
        return "Error cannot stitch images"

    (matches, homography_matrix, status) = M

    result = image_stitching.blending_smoothing(
        query_photo, train_photo, homography_matrix
    )
    # mapped_image = cv2.drawMatches(train_photo, keypoints_train_image, query_photo, keypoints_query_image, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    mapped_float_32 = np.float32(mapped_feature_image)
    result_float32 = np.float32(result)
    result_rgb = cv2.cvtColor(result_float32, cv2.COLOR_BGR2RGB)
    mapped_feature_image_rgb = cv2.cvtColor(mapped_float_32, cv2.COLOR_BGR2RGB)
    
    return result_rgb, mapped_feature_image_rgb


if __name__ == "__main__":
    try:
        query_image = sys.argv[1]
        train_image = sys.argv[2]

        def read_images(image):
            photo = cv2.imread(image)
            photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

            return photo

        query_image = read_images(query_image)
        train_image = read_images(train_image)

        result = forward(query_photo=query_image, train_photo=train_image)
        cv2.imwrite("output/panorama_image.jpg", result)
    except IndexError:
        print("Please input atleast two source images")