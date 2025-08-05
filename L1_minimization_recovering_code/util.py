"""
Functions I used in the past, which is pretty helpful I think
Import this class, and you can use all functions.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def create_kernel(size : int) :
    return np.ones((size, size), np.uint8)

def read_part_image_upper(percentage : int, pict) -> np.ndarray:
    """
    Extract the upper portion of an image based on a given percentage.

    This function takes an image and returns its upper region,
    determined by dividing the image height by the specified percentage.
    For example, if percentage=3, it returns the top third of the image.

    :param percentage: the fraction of the image to keep from the top.
    :param pict: image array
    :return : a Numpy array represents the extracted upper part of the image
    """
    height, width = pict.shape[:2]

    upper_third = height // percentage
    upper_region = pict[0:upper_third, :].copy()

    return upper_region

def show(pict : np.ndarray, pict_name : str = 'pict',) -> None :
    """
    This function aims to simplify the process of displaying images.
    :param pict_name: Name of the image.
    :param pict: the image array
    :return: (no return value)
    """
    cv.imshow(pict_name, pict)
    while True:
        key = cv.waitKey(100)
        if cv.getWindowProperty(pict_name, cv.WND_PROP_VISIBLE) < 1:
            break
        if key != -1:
            break
    cv.destroyAllWindows()

def show_histgram(pict : np.ndarray) -> None:
    """
    This function displays the histogram of pixel values in the image,
    helping us understand the distribution of pixel intensities.
    :param pict: image arrary
    :return: (No return value)
    """
    hist = cv.calcHist([pict], [0], None, [256], [0, 256])

    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def equalized_color_pict(pict) :
    """
    This function is used to equalize a color image and enhance its contrast.
    :param pict: original image array, must be a colorful image.
    :return: a NumPy array with enhanced contrast.
    """
    # Separate color channel
    b, g, r = cv.split(pict)

    # Equalized histogram in each channel
    b_eq = cv.equalizeHist(b)
    g_eq = cv.equalizeHist(g)
    r_eq = cv.equalizeHist(r)

    # Merge back
    equalized_image = cv.merge([b_eq, g_eq, r_eq])

    return equalized_image


def unpickle(file_path : str):
    """
    This function is designed for getting data from CIFAR-100

    It gets the file path of CIFAR-100 and return image data
    in the dataset
    """
    import pickle
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data']

def get_test_image(file_path : str) :
    """
    This function is designed for converting image data in the CIFAR-100
    to Numpy arrays

    It gets file path of CIFAR-100 and then return the images array
    """
    image_data = unpickle(file_path)
    test_images = []

    for image_array in image_data :
        test_image = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
        test_images.append(test_image)

    return test_images












