"""
Where our works begin
"""
from L1_minimization import *
from util import *

def get_random_coordinates(width, height, percentage):
    """
    This function is designed for randomly create coordinate base
    on the shape of image

    It receives the width and height of a Numpy array(image) and
    how many percentage of image need to be missing
    Finally, it returns coordinates of missing value
    """
    image_destroy_in_different_case = []

    max_num = int(percentage * width * height)

    indices = np.random.choice(width * height, size=max_num, replace=False)

    cols_positions = indices // width
    rows_positions = indices % width

    coordinates = np.stack((rows_positions, cols_positions), axis=1)

    image_destroy_in_different_case.append(coordinates)

    return image_destroy_in_different_case

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






if __name__ == "__main__"  :
    # read grayscale image
    image = cv.imread("../test_images/smallTriangle.png", cv.IMREAD_GRAYSCALE)

    # get image with missing values
    destroy_image, missing_coords = random_destroy_to_image(image)

    g_result, accuracy = solve_l1_minimization(destroy_image, missing_coords)

    print(g_result)










