import model as m
import sys
import cv2

def main(image_dir_list):
    """Main function of the Repository.
    Takes in list of image dir, runs the complete image stitching pipeline
    to create and export a panoramic image in the /outputs/ folder.

    Args:
        image_dir_list (List): List of image dirs passed in cmdline
    """

    images_list, no_of_images = m.read(image_dir_list)
    result, mapped_image = m.recurse(images_list, no_of_images)
    cv2.imwrite("src/feature_based/output/panorama_image.jpg", result)
    cv2.imwrite("src/feature_based/output/mapped_image.jpg", mapped_image)

    print(f"Panoramic image saved at: outputs/panorama_image.jpg")


if __name__ == "__main__":
    image_list = []
    for i in range(1, len(sys.argv)):
        image_list.append(sys.argv[i])
    main(image_list)

# code chạy thử
# python src/feature_based/Paronama.py src/feature_based/input/room/room01.jpeg src/feature_based/input/room/room02.jpeg