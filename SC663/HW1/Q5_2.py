import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Temperature measurements
t = np.arange(1, 25)
y = np.array([75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52, 50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72], dtype=float)

# Normalize the sampling instants
s = (t - 12.5) / 11.5

# Construct the Vandermonde matrix
n = 24
Phi = np.vander(s, N=n, increasing=True)

# Interpolating polynomial
x_interp = np.linalg.solve(Phi, y)

# Values of lambda
lambdas = [1e-4, 1e-2, 1e-1, 1, 10, 100]

# Storage for coefficient vectors
ridge_coefficients = {}
sparse_coefficients = {}

# Solve ridge and sparse polynomial problems
for lam in lambdas:
    x_ridge = cp.Variable(n)
    ridge_problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(Phi @ x_ridge - y) + 0.5 * lam * cp.sum_squares(x_ridge)))
    ridge_problem.solve()

    if x_ridge.value is not None:
        ridge_coefficients[lam] = x_ridge.value.copy()

    x_sparse = cp.Variable(n)
    sparse_problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(Phi @ x_sparse - y) + lam * cp.norm1(x_sparse)))
    sparse_problem.solve()

    if x_sparse.value is not None:
        sparse_coefficients[lam] = x_sparse.value.copy()

# Plotting points
s_plot = np.linspace(-1, 1, 500)
Phi_plot = np.vander(s_plot, N=n, increasing=True)

# Plot interpolating polynomial
y_interp = Phi_plot @ x_interp

plt.figure(figsize=(12, 7))
plt.scatter(s, y, label="Temperature measurements")
plt.plot(s_plot, y_interp, label="Interpolating polynomial")
plt.xlabel("Normalized sampling instant")
plt.ylabel("Temperature")
plt.title("Interpolating Polynomial")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot ridge and sparse fits for each lambda
for lam in lambdas:
    plt.figure(figsize=(12, 7))
    plt.scatter(s, y, label="Temperature measurements")

    if lam in ridge_coefficients:
        y_ridge = Phi_plot @ ridge_coefficients[lam]
        plt.plot(s_plot, y_ridge, label=f"Ridge, lambda = {lam}")

    if lam in sparse_coefficients:
        y_sparse = Phi_plot @ sparse_coefficients[lam]
        plt.plot(s_plot, y_sparse, "--", color="orange", label=f"Sparse, lambda = {lam}")

    plt.xlabel("Normalized sampling instant")
    plt.ylabel("Temperature")
    plt.title(f"Polynomial Fits for lambda = {lam}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot coefficient vectors for each lambda
for lam in lambdas:
    plt.figure(figsize=(12, 7))
    x_axis = np.arange(n)

    if lam in ridge_coefficients:
        plt.plot(x_axis, ridge_coefficients[lam], label=f"Ridge, lambda = {lam}")

    if lam in sparse_coefficients:
        plt.plot(x_axis, sparse_coefficients[lam], "--", color="orange", label=f"Sparse, lambda = {lam}")

    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient value")
    plt.title(f"Coefficient Vectors for lambda = {lam}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()