"""
Copy code from Professor Alex's code.
For comparison
"""
import numpy as np
import cvxpy as cp
from PIL import Image
import matplotlib.pyplot as plt
import random


def jpg_to_grayscale_array(image_path):
    """
    Takes a .jpg image file and returns its grayscale version as a 2D NumPy array.

    Parameters:
        image_path (str): The path to the .jpg image file.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the grayscale image.
    """
    # Open the image using Pillow
    image = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    return image_array

def display_grayscale_image(image_array):
    """
    Takes a 2D NumPy array and displays it as a grayscale image.

    Parameters:
        image_array (numpy.ndarray): A 2D NumPy array with pixel intensity values in the range [0, 255].
    """
    # Ensure the image array is of the correct dtype (uint8)
    image_array = np.asarray(image_array, dtype=np.uint8)

    # Convert the NumPy array to a Pillow image
    image = Image.fromarray(image_array)

    # Display the image using Matplotlib
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Hide axes for a cleaner image display
    plt.show()

# @profile # for estimating the time complexity
# L1 optimizer for 2D arrays
def L1optimizer_2D(input_array, M):
    """
    Optimize a function g to minimize the sum of absolute values of its 2D Fourier coefficients
    while ensuring g(x, y) equals f(x, y) for indices not in the subset M.

    Args:
        input_array (numpy.ndarray): 2D input array of real numbers.
        M (set of tuple): Set of (x, y) indices where g can deviate from f.

    Returns:
        tuple:
            - numpy.ndarray: The optimized 2D array g.
            - float: The accuracy measurement.
    """
    if not isinstance(input_array, np.ndarray) or input_array.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Dimensions of the input array
    rows, cols = input_array.shape

    # Convert input to a NumPy array
    f = np.array(input_array, dtype=float)

    # 2D Fourier transform matrix
    omega_row = np.exp(-2j * np.pi / rows)
    omega_col = np.exp(-2j * np.pi / cols)

    F_row = np.array([[omega_row ** (x * m) for m in range(rows)] for x in range(rows)]) / np.sqrt(rows)
    F_col = np.array([[omega_col ** (y * n) for n in range(cols)] for y in range(cols)]) / np.sqrt(cols)

    # Optimization variables
    g = cp.Variable((rows, cols), value=f)

    # Compute the 2D Fourier transform of g
    g_fft = F_row @ g @ F_col.T

    # Constraints: g(x, y) = f(x, y) for (x, y) not in M
    constraints = [g[x, y] == f[x, y] for x in range(rows) for y in range(cols) if (x, y) not in M]

    # Objective: Minimize sum of absolute values of 2D Fourier coefficients
    objective = cp.Minimize(cp.sum(cp.abs(g_fft)))

    # Problem definition
    problem = cp.Problem(objective, constraints)

    try:
        # Solve the problem using SCS solver
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status != cp.OPTIMAL:
            raise ValueError("Optimization failed to find a solution.")

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None  # Handle failure appropriately

    # Extract the optimized g(x, y)
    g_result = g.value


    # Check if the result is None before proceeding
    if g_result is None:
        print("Optimization did not return a valid result.")
        return None, None

    # Compute accuracy measurement
    numerator = np.sum([abs(f[x, y] - g_result[x, y]) for x, y in M])
    denominator = np.sum([abs(f[x, y]) for x, y in M])
    accuracy = numerator / denominator if denominator != 0 else float('inf')

    # Return g(x, y) and accuracy measurement
    return g_result, accuracy

# @profile # for estimating the time complexity
# L1 optimizer followed by Fourier pruning for 2D arrays
def L1prunedSTDoptimizer_2D(input_array, M, c):
    """
    Optimizes a function g using L1 optimization and then prunes its 2D Fourier coefficients
    based on a threshold defined as c standard deviations above the mean magnitude.

    Args:
        input_array (numpy.ndarray): 2D input array of real numbers.
        M (set of tuple): Set of (x, y) indices where g can deviate from f.
        c (float): Scaling factor for the standard deviation threshold.

    Returns:
        tuple:
            - numpy.ndarray: The optimized and pruned 2D array H.
            - float: The accuracy measurement.
    """
    # 1. Obtain g(x, y) and accuracy from L1optimizer_2D
    g, accuracy = L1optimizer_2D(input_array, M)
    if g is None:
        return None, None  # Handle failure from L1optimizer

    # 2. Compute 2D Fourier transform of g(x, y)
    rows, cols = g.shape
    g_fft = np.fft.fft2(g) / np.sqrt(rows * cols)  # Normalize by sqrt(rows * cols)


    # 3. Compute mean and standard deviation of absolute values of Fourier coefficients
    abs_g_fft = np.abs(g_fft)
    mean_abs_g_fft = np.mean(abs_g_fft)
    std_abs_g_fft = np.std(abs_g_fft)

    # 4. Create h(x, y) by pruning Fourier coefficients based on threshold
    threshold = mean_abs_g_fft + c * std_abs_g_fft
    h_fft = np.where(abs_g_fft >= threshold, g_fft, 0)
    h = np.fft.ifft2(h_fft * np.sqrt(rows * cols)).real  # Inverse FFT, keep real part

    # 5. Create H(x, y) by combining h(x, y) and input_array
    H = np.array(input_array, dtype=float)  # Initialize with input_array

    for x, y in M:
        H[x, y] = h[x, y]  # Replace values at indices in M with h(x, y)


    # 6. Compute accuracy of H(x, y)
    numerator = np.sum([abs(input_array[x, y] - H[x, y]) for x, y in M])
    denominator = np.sum([abs(input_array[x, y]) for x, y in M])
    accuracy_H = numerator / denominator if denominator != 0 else float('inf')



    return H, accuracy_H



def generate_random_coordinates(array_shape, m):
    """
    Generates a random set of coordinates from a 2D numpy array.

    Parameters:
        array_shape (tuple): The shape of the 2D numpy array (rows, cols).
        m (int): The number of random coordinates to generate.

    Returns:
        set of tuple: A set of m random (row, col) coordinates.
    """
    rows, cols = array_shape

    # Ensure m is not larger than the total number of elements in the array
    if m > rows * cols:
        raise ValueError("m cannot be larger than the total number of elements in the array.")

    # Generate all possible coordinates
    all_coordinates = [(i, j) for i in range(rows) for j in range(cols)]

    # Randomly sample m unique coordinates
    random_coordinates = random.sample(all_coordinates, m)

    # Convert to a set of tuples
    return set(random_coordinates)


# Example usage
# array_shape = (4, 5)  # Shape of the array (4 rows, 5 columns)
array_shape = (10, 10)
m = 50  # Number of random coordinates to generate

# random_coords = generate_random_coordinates(array_shape, m)
# print("Random Coordinates:", random_coords)

if __name__ == "__main__" :
    # Example usage
    image_path = '../test_images/smallTriangle.png'  # Replace with your image path
    # image_path = '../../Desktop/STEMFall2025/STEMFall2025/img.png'  # Replace with your image path

    grayscale_image_array = jpg_to_grayscale_array(image_path)

    # print(grayscale_image_array)  # Prints the 2D numpy array

    # Example usage:
    # Create a sample 2D NumPy array (grayscale image)
    # height, width = 100, 100
    # sample_image_array = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

    # Call the function to display the image
    # display_grayscale_image(sample_image_array)
    # display_grayscale_image(grayscale_image_array)
    # display_grayscale_image(sample_image_array), display_grayscale_image(grayscale_image_array)

    # Example usage
    # array_shape = (4, 5)  # Shape of the array (4 rows, 5 columns)
    # array_shape = (10, 10)
    # m = 50  # Number of random coordinates to generate

    # random_coords = generate_random_coordinates(array_shape, m)
    # print("Random Coordinates:", random_coords)


    # Define a 2D input array
    # input_array = np.array([
    #    [1.0, 2.0, 3.0],
    #    [4.0, 5.0, 6.0],
    #    [7.0, 8.0, 9.0]
    # ])

    input_array = np.array([
        [10, 2, 1, 1, 10, 10, 1, 1, 2, 10],
        [1, 10, 1, 1, 1, 1, 1, 1, 10, 1],
        [1, 1, 10, 1, 1, 1, 1, 10, 1, 1],
        [1, 1, 1, 10, 1, 1, 10, 1, 1, 1],
        [10, 1, 1, 1, 10, 10, 1, 1, 1, 10],
        [10, 1, 1, 1, 10, 10, 1, 1, 1, 10],
        [1, 1, 1, 10, 1, 1, 10, 1, 1, 1],
        [1, 1, 10, 1, 1, 1, 1, 10, 1, 1],
        [1, 10, 1, 1, 1, 1, 1, 1, 10, 1],
        [10, 2, 1, 1, 10, 10, 1, 1, 2, 10],
    ])


    input_array = grayscale_image_array

    # Define the set of indices (x, y) where g can deviate from f
    # M = {(0, 1), (1, 2), (2, 0)}
    # M = random_coords

    # Define the threshold scaling factor
    c = 1.0

    # Use the L1prunedSTDoptimizer_2D function

    a = 0
    for i in range(100):

        random_coords = generate_random_coordinates(array_shape, m)

        # J = random_coords
        J = {(1,0), (2,0), (3,0), (4,0), (5,0), (5,1), (5,2), (5,3), (5,4)}

        H, accuracy_H = L1prunedSTDoptimizer_2D(input_array, J, .2)
        # a += 100 * (1 - accuracy_H)

        a = 1 - accuracy_H

        # print(a / 100)
        print(a)

        # Display the results
        print("Optimized and Pruned Array (H):")
        print(H)

        print("\nAccuracy of the Optimized Array:")
        print(100 * (1 - accuracy_H))

        print( 20 * "*")



