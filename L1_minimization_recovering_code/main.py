"""
Where our works begin
"""
from L1_minimization import *
from util import *
from L1 import *
import time

def get_random_coordinates(width, height, percentage):
    """
    This function is designed for randomly create coordinate base
    on the shape of image

    It receives the width and height of a Numpy array(image) and
    how many percentage of image need to be missing
    Finally, it returns coordinates of missing value
    """
    image_with_missing_values = []

    max_num = int(percentage * width * height)

    indices = np.random.choice(width * height, size=max_num, replace=False)

    cols_positions = indices // width
    rows_positions = indices % width

    coordinates = np.stack((rows_positions, cols_positions), axis=1)

    image_with_missing_values.append(coordinates)

    return image_with_missing_values

def random_destroy_to_image(image_array : np.ndarray, missing_value : int = 125,
                            percentage : float = 0.1) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    This function is design for creating different cases of destroy to the image.
    (That is, adding different numbers of missing values to the image)

    It receives the image Numpy array, an arbitrary setting missing values and a
    float percentage value to imply how many missing values need to be added.

    It returns a list of Numpy array, which are destroyed image.
    """

    # get the width and length of image
    width, height = image_array.shape

    # generate random coordinates that need to be missing value
    missing_coordinates = get_random_coordinates(width, height, percentage)[0]

    # create a mask to show where are missing values
    mask_image = image_array.copy()

    # replace missing values with arbitrary values
    for x, y in missing_coordinates:
        mask_image[x][y] = missing_value

    return mask_image, missing_coordinates

def get_report_name(t_file_names : bytes) :
    """
    This function is designed for generating report name from
    the CIFAR-10 dataset

    It gets the filename from CIFAR-10 dataset, which is a bytes
    string, converting the bytes into str and then return the final
    names.
    """
    file_name = t_file_names.decode('utf-8')

    name_len = len(file_name)

    report_name = file_name[:name_len - 4] + "_report" + file_name[-4:]

    return report_name



# Multiple image test

if __name__ == "__main__"  :

    test_images, file_names = get_test_image("../cifar-10-batches-py/test_batch")

    for index in range(10) :
        t_image = test_images[index]
        t_file_name = get_report_name(file_names[index])
        # read grayscale image
        original_image = cv.cvtColor(t_image, cv.COLOR_RGB2GRAY)


        # get image with missing values
        destroy_image, missing_coords = random_destroy_to_image(original_image, missing_value=0, percentage=0.3)
        missing_set = set(map(tuple, missing_coords))

        # Our way to achieve L1 minimization, basically the same
        # but different in doing Fourier transform

        start = time.time()
        recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, .2)
        end = time.time()
        print(f"Our total time used: {end - start} seconds.")

        # Use the L1prunedSTDoptimizer_2D function
        start_a = time.time()
        recovered_image_a, accuracy_H_a = L1prunedSTDoptimizer_2D(destroy_image, missing_set, .2)
        end_a = time.time()
        print(f"Alex's total time used: {end_a - start_a} seconds.")

        generate_report(original_image, destroy_image, recovered_image_a,
                        t_file_name,
                        save_path="../reports_alex")

        generate_report(original_image, destroy_image, recovered_image,
                        t_file_name,
                        save_path="../reports")



# # Single image test
# if __name__ == "__main__" :
#     # test_images = get_test_image("../cifar-10-batches-py/test_batch")
#
#     # image = cv.cvtColor(test_images[0], cv.COLOR_RGB2GRAY)
#
#     # show(image)
#
#     # read grayscale image
#     original_image = cv.imread("../test_images/smallTriangle.png", cv.IMREAD_GRAYSCALE)
#     #
#     destroy_image, missing_coords = random_destroy_to_image(original_image, missing_value=125, percentage=0.3)
#     missing_set = set(map(tuple, missing_coords))
#
#     # Our way to achieve L1 minimization, basically the same
#     # but different in doing Fourier transform
#     start = time.time()
#     recovered_image, accuracy = do_image_recovery(destroy_image, missing_set, .2)
#     end = time.time()
#     print(f"Our total time used: {end - start} seconds.")
#     print(f"\n H :{recovered_image}; Accuracy: {accuracy}")
#
#     generate_report(original_image, destroy_image, recovered_image,
#                     "smallTriangle_report.png")
#
#     # # Use the L1prunedSTDoptimizer_2D function
#     # start_a = time.time()
#     # recovered_image_a, accuracy_H_a = L1prunedSTDoptimizer_2D(destroy_image, J, .2)
#     # end_a = time.time()
#     # print(f"Alex's total time used: {end_a - start_a} seconds.")
#     #
#     # generate_report(original_image, destroy_image, recovered_image_a,
#                     "smallTriangle_report.png", save_path="../reports_alex")












