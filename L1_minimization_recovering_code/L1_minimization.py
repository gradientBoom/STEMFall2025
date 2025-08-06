from .util import *
import cvxpy as cp

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

def build_inverse_Fourier_base_matrix(length : int) -> np.ndarray :
    """
    This function is designed for building Fourier transform base
    matrix base on this article https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/

    The same as how to build Fourier transform, it takes the length
    of  a 1D-array and return its inverse Fourier Transform base
    Matrix
    """
    omega_input = np.exp(2j * np.pi / length)

    F_input_hat = np.array([[omega_input ** (x * m) for m in range(length)]
                        for x in range(length)])

    return F_input_hat


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

def do_inverse_Fourier_transform(input_array) :
    """
    This function is designed for applying inverse Fourier transformation to input
    array,
    It receives a 2D Numpy array, then convert it to a normal 2D Numpy array
    """
    rows, cols = input_array.shape

    F_rows_hat = build_inverse_Fourier_base_matrix(rows)
    F_cols_hat = build_inverse_Fourier_base_matrix(cols)

    g_fft_hat = F_rows_hat @ input_array @ F_cols_hat / (rows * cols)

    return np.real(g_fft_hat).round().astype(np.uint8)

# @profile # for estimating the time complexity
def solve_l1_minimization(image_array : np.ndarray, missing_coords) :
    rows, cols = image_array.shape

    # Optimization variables
    g = cp.Variable((rows, cols), value=image_array)

    # Compute the 2D Fourier transform of g
    g_fft = do_Fourier_transform(g)

    # Constraints: g(x, y) = f(x, y) for (x, y) not in missing_coords.
    constraints = [g[x, y] == image_array[x, y] for x in range(rows) for y in range(cols) if (x, y) not in missing_coords]

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
    numerator = np.sum([abs(image_array[x, y] - g_result[x, y]) for x, y in missing_coords])
    denominator = np.sum([abs(image_array[x, y]) for x, y in missing_coords])
    accuracy = numerator / denominator if denominator != 0 else float('inf')

    # Return g(x, y) and accuracy measurement
    return g_result, accuracy

# @profile # for estimating the time complexity
def do_image_recovery(image_with_missing_values, missing_coordinates, threshold) :
    """
    This function is designed for doing image recovery by optimizing a function
    using L1 minimization

    It receives the image with missing values and a missing set
    """
    g, accuracy = solve_l1_minimization(image_with_missing_values, missing_coordinates)

    rows, cols = g.shape
    g_fft = np.fft.fft2(g) / np.sqrt(rows * cols)

    abs_g_fft = np.abs(g_fft)
    mean_abs_g_fft = np.mean(abs_g_fft)
    std_abs_g_fft = np.std(abs_g_fft)

    threshold_values = mean_abs_g_fft + threshold * std_abs_g_fft
    h_fft = np.where(abs_g_fft >= threshold_values, g_fft, 0)
    h = np.fft.ifft2(h_fft * np.sqrt(rows * cols)).real

    H = np.array(image_with_missing_values, dtype=float)

    for x, y in missing_coordinates :
        H[x, y] = h[x, y]

    # 6. Compute accuracy of H(x, y)
    numerator = np.sum([abs(image_with_missing_values[x, y] - H[x, y]) for x, y in missing_coordinates])
    denominator = np.sum([abs(image_with_missing_values[x, y]) for x, y in missing_coordinates])
    accuracy_H = numerator / denominator if denominator != 0 else float('inf')


    return H, accuracy_H