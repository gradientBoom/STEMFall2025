"""
Functions I used in the past, which is pretty helpful I think
Import this class, and you can use all functions.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def build_Fourier_base_matrix(length : int) -> np.ndarray :
    """
    This function is designed for building Fourier transform base
    matrix according to Professor Alex's code

    It gets the length of a 1D-array and return its Fourier transformation
    base matrix
    """
    omega_input = np.exp(-2j * np.pi / length)

    # F_input = np.array([[omega_input ** (x * m) for m in range(length)]
    #                     for x in range(length)]) / np.sqrt(length)
    F_input = np.array([[omega_input ** (x * m) for m in range(length)]
                        for x in range(length)])

    return F_input

def do_Fourier_transform(input_array):
    """
    This function is designed for applying Fourier transformation to input array
    It get a 2D Numpy array, which should be the image array then turn it into
    a Fourier 2D array, still Numpy
    """
    rows, cols = input_array.shape

    F_row = build_Fourier_base_matrix(rows)
    F_col = build_Fourier_base_matrix(cols)

    # Compute the 2D Fourier transform of g
    g_fft = F_row @ input_array @ F_col.T

    return g_fft

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
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def get_test_image(file_path : str) :
    """
    This function is designed for converting image data in the CIFAR-10
    to Numpy arrays

    It gets file path of CIFAR-10 and then return the images array
    """
    source_data_dir = unpickle(file_path)

    file_names = source_data_dir[b'filenames']
    image_data = source_data_dir[b'data']
    test_images = []

    for image_array in image_data :
        test_image = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
        test_images.append(test_image)

    return test_images, file_names

def generate_report(original_image : np.ndarray,
                    masked_image : np.ndarray,
                    recovered_image : np.ndarray,
                    save_file_name : str,
                    save_path : str = "../reports",
                    show_report : bool = False) -> None:
    """
    This function is designed for generate reports of image recovery

    It receives Numpy array of original, masked, recovered
    images and the save filename. Then it saves graph reports under the
    save_path folders.
    It can directly show report for each image with the parameter:
    show_report = True
    """

    fft_orig = np.abs(do_Fourier_transform(original_image * 255))
    fft_masked = np.abs(do_Fourier_transform(masked_image * 255))
    fft_recovered = np.abs(do_Fourier_transform(recovered_image * 255))

    psnr_value, ssim_value = get_accuracy(original_image, recovered_image)

    plt.figure(figsize=(12, 4))

    plt.subplot(2,4,1)
    plt.title("Original")
    plt.imshow(original_image, cmap='gray')

    plt.subplot(2,4,2)
    plt.title("Masked")
    plt.imshow(masked_image, cmap='gray')

    plt.subplot(2,4,3)
    plt.title("Recovered")
    plt.imshow(recovered_image, cmap='gray')

    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.text(0.1, 0.5, f"PSNR: {psnr_value:.2f}\nSSIM: {ssim_value:.4f}")

    plt.subplot(2, 4, 5)
    plt.title("FFT Original")
    plt.imshow(np.log1p(fft_orig), cmap='gray')
    plt.subplot(2, 4, 6)
    plt.title("FFT Masked")
    plt.imshow(np.log1p(fft_masked), cmap='gray')
    plt.subplot(2, 4, 7)
    plt.title("FFT Recovered")
    plt.imshow(np.log1p(fft_recovered), cmap='gray')
    plt.tight_layout()

    if show_report :
        plt.show()

    plt.savefig(save_path + "/" + save_file_name, bbox_inches='tight')

def get_accuracy(original_image : np.ndarray, recovered_image : np.ndarray) :
    """
    This function is designed for calculating the accuracy of recovering image by
    calculating PSNR and SSIM value

    It receives original images and recovered images, then return the accuracy of
    recovered image
    """
    psnr_val = psnr(original_image, recovered_image, data_range=255)
    ssim_val = ssim(original_image, recovered_image, data_range=255)

    return tuple([psnr_val, ssim_val])












