from torch.backends.mkl import verbose

from util import *
import cvxpy as cp

def build_Fourier_base_matrix(length : int) -> np.ndarray :
    """
    This function is designed for building Fourier transform base
    matrix according to Professor Alex's code

    It gets the length of a 1D-array and return its Fourier transformation
    base matrix
    """
    omega_input = np.exp(-2j * np.pi / length)

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


def do_Fourier_transform(input_array : np.ndarray):
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

def solve_l1_minimization(image_array : np.ndarray, missing_coords) :
    rows, cols = image_array.shape

    g = cp.Variable((rows, cols), value=image_array)
    g_fft = do_Fourier_transform(image_array)


    constraints = [g[x, y] == image_array[x, y] for x in range(rows) for y in range(cols) if (x, y) not in missing_coords]

    objective = cp.Minimize(cp.sum(cp.abs(g_fft)))

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









