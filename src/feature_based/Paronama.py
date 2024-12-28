import model as m
import sys
import cv2
import time

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

    

def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution : {execution_time:.4f} seconds")
    return result

if __name__ == "__main__":
    import sys

    # Change your path
    default_image_list = [
        "data/UDIS-D/testing/input1/000002.jpg",
        "data/UDIS-D/testing/input2/000002.jpg"
    ]

    image_list = []
    for i in range(1, len(sys.argv)):
        image_list.append(sys.argv[i])

    image_list.extend(default_image_list)

    main(image_list)
