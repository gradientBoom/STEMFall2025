import numpy as np

from util import *

def build_Fourier_base_matrix(length : int) -> np.ndarray :

    omega_input = np.exp(-2j * np.pi / length)

    F_input = np.array([[omega_input ** (x * m) for m in range(length)]
                        for x in range(length)])

    return F_input

def build_inverse_Fourier_base_matrix(length : int) -> np.ndarray :
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

def solve_l1_minimization(input_array) :
    pass


# def something() -> None:
#     # Constraints: g(x, y) = f(x, y) for (x, y) not in M
#     constraints = [g[x, y] == input_array[x,y] for x in range(rows) for y in range(cols) if(x, y) not in M]
#
#     objective = cp.Minimize(cp.sum(cp.abs(g_fft)))
#
#
#     problem = cp.Problem(objective, constraints)
#
#     try :
#
#         problem.solve(solver=cp.SCS, verbose=False)
#
#         if problem.status != cp.OPTIMAL :
#             raise ValueError("Optimization failed to find a solution")
#
#     except Exception as e :
#         print(f"Error occurred : {e}")
#         return None
#
#     g_result = g.value
#
#     if g_result is None :
#         print("Didn't return a valid result.")
#         return None
#
#     return g_result






