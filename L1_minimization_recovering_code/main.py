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






# if __name__ == "__main__"  :
#
#     test_images = get_test_image("../cifar-100-python/test")
#
#     for t_image in test_images :
#         # read grayscale image
#         image = cv.cvtColor(t_image, cv.COLOR_RGB2GRAY)
#         show(image, "Original")
#
#         # get image with missing values
#         destroy_image, missing_coords = random_destroy_to_image(image, missing_value=0, percentage=0.1)
#
#         J = set(map(tuple, missing_coords))
#
#         # Our way to achieve L1 minimization, basically the same
#         # but different in doing Fourier transform
#
#         start = time.time()
#         H, accuracy = do_image_recovery(destroy_image, J, .2)
#         end = time.time()
#         print(f"Our total time used: {end - start} seconds.")
#
#         # Use the L1prunedSTDoptimizer_2D function
#
#         start_a = time.time()
#         H_a, accuracy_H_a = L1prunedSTDoptimizer_2D(destroy_image, J, .2)
#         end_a = time.time()
#         print(f"Alex's total time used: {end_a - start_a} seconds.")
#
#         show(H, "Recovered")
#         print("\n")

if __name__ == "__main__" :



    # read grayscale image
    image = cv.imread("../test_images/smallTriangle.png", cv.IMREAD_GRAYSCALE)

    destroy_image, missing_coords = random_destroy_to_image(image, missing_value=125, percentage=0.5)
    J = set(map(tuple, missing_coords))

    # Our way to achieve L1 minimization, basically the same
    # but different in doing Fourier transform
    start = time.time()
    H, accuracy = do_image_recovery(destroy_image, J, .2)
    end = time.time()
    print(f"Our total time used: {end - start} seconds.")
    print(f"\n H :{H}; Accuracy: {accuracy}")

    show(H,"Our Recovery")

    # Use the L1prunedSTDoptimizer_2D function
    start_a = time.time()
    H_a, accuracy_H_a = L1prunedSTDoptimizer_2D(destroy_image, J, .2)
    end_a = time.time()
    print(f"Alex's total time used: {end_a - start_a} seconds.")
    print(f"\n H :{H_a}; Accuracy: {accuracy_H_a}")

    show(H_a,"Alex Recovery")












