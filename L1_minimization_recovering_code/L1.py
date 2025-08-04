import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def check_ones(x, A):
    return 1 if sum(x) in A else 0

def int_to_binvec(i, N):
    return [int(b) for b in bin(i)[2:].zfill(N)]

def fwht(a):
    """Fast Walsh–Hadamard Transform (in-place)."""
    a = a.copy()
    h = 1
    n = len(a)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def hadamard_matrix(n):
    if n == 1:
        return np.array([[1.0]])
    H = hadamard_matrix(n // 2)
    return np.block([[H, H], [H, -H]])

def boolean_function_l1_imputation_fwht(N, A, delta, seed=0, plot_fourier=False):
    np.random.seed(seed)

    num_points = 2 ** N
    X = np.array([int_to_binvec(i, N) for i in range(num_points)])
    f_vals = np.array([check_ones(x, A) for x in X], dtype=np.float64)

    # Optionally compute and plot Fourier transform (continuous)
    if plot_fourier:
        fourier_coeffs = fwht(f_vals) / num_points
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(num_points), fourier_coeffs, linewidth=2)
        plt.title("Fourier Transform of check_ones")
        plt.xlabel("Index")
        plt.ylabel("Coefficient Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    indices = np.arange(num_points)
    np.random.shuffle(indices)

    num_masked = int(delta * num_points)
    masked_indices = indices[:num_masked]
    known_indices = np.setdiff1d(indices, masked_indices)

    # Walsh-Hadamard matrix (normalized)
    H = hadamard_matrix(num_points) / num_points
    u_hat = cp.Variable(num_points)

    # Reconstructed signal (inverse FWHT)
    u_vals = H @ u_hat

    # Constraint: must match known values
    constraints = [u_vals[i] == f_vals[i] for i in known_indices]
    objective = cp.Minimize(cp.norm1(u_hat))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Predict using solution
    u_vals_rec = u_vals.value
    g_vals = np.round(u_vals_rec).astype(int)
    g_vals = np.clip(g_vals, 0, 1)

    errors = np.sum(g_vals[masked_indices] != f_vals[masked_indices])
    error_rate = errors / len(masked_indices)

    return errors, len(masked_indices), error_rate

if __name__ == "__main__" :
    # Import the l1-minimization code above
    # Then run the following usage example

    N = 9  # Input dimension
    A = {0, 1}  # Allowed Hamming weights
    delta = 0.5  # Fraction of data to mask
    seed = 42

    # Run the imputation with Fourier plot enabled
    errors, total_masked, error_rate = boolean_function_l1_imputation_fwht(N, A, delta, seed, plot_fourier=True)

    print("🧮 L1 Minimization via FWHT")
    print(f"Masked inputs: {total_masked}")
    print(f"Mismatches: {errors}")
    print(f"Error rate: {error_rate:.2%}")
