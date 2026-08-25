import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

# Temperature measurements
t = np.arange(1, 25)
y = np.array([75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52,
              50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72])

# Normalize sampling instants
s = (t - 12.5) / 11.5

# Vandermonde matrix for a polynomial of order 23
Phi = np.vander(s, 24, increasing=True)

# Regularization parameters
lambdas = [1e-4, 1e-2, 1e-1, 1, 10, 100]

ridge_errors = []
sparse_errors = []
ridge_nnz = []
sparse_nnz = []

for lam in lambdas:
    # Ridge regression
    x = cp.Variable(24)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(Phi @ x - y) +
                    0.5 * lam * cp.sum_squares(x))
    )
    problem.solve()
    ridge_x = x.value

    # Sparse regression
    x = cp.Variable(24)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(Phi @ x - y) +
                    lam * cp.norm1(x))
    )
    problem.solve()
    sparse_x = x.value

    # Relative fitting errors
    ridge_error = np.linalg.norm(Phi @ ridge_x - y) / np.linalg.norm(y)
    sparse_error = np.linalg.norm(Phi @ sparse_x - y) / np.linalg.norm(y)

    # Number of nonzero coefficients
    ridge_count = np.count_nonzero(np.abs(ridge_x) > 1e-6)
    sparse_count = np.count_nonzero(np.abs(sparse_x) > 1e-6)

    ridge_errors.append(ridge_error)
    sparse_errors.append(sparse_error)
    ridge_nnz.append(ridge_count)
    sparse_nnz.append(sparse_count)

    print(f"lambda = {lam}")
    print(f"Ridge  : E(x) = {ridge_error:.6f}, ||x||_0 = {ridge_count}")
    print(f"Sparse : E(x) = {sparse_error:.6f}, ||x||_0 = {sparse_count}")
    print()

# Create output directory
os.makedirs("images", exist_ok=True)

# Plot relative fitting error
plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, ridge_errors, 'o-', label='Ridge')
plt.semilogx(lambdas, sparse_errors, 'o--', label='Sparse')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Relative fitting error $E(x)$')
plt.title('Relative Fitting Error vs. Regularization')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("images/Q5_3_fitting_error.png", dpi=300)
plt.show()

# Plot sparsity
plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, ridge_nnz, 'o-', label='Ridge')
plt.semilogx(lambdas, sparse_nnz, 'o--', label='Sparse')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Number of nonzero coefficients')
plt.title('Sparsity vs. Regularization')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("images/Q5_3_sparsity.png", dpi=300)
plt.show()