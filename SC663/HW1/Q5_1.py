import numpy as np
import cvxpy as cp

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
interp_error = np.linalg.norm(Phi @ x_interp - y)

print("INTERPOLATING POLYNOMIAL")
print("Coefficient vector:")
print(x_interp)
print(f"\nFitting residual ||Phi x - y||_2 = {interp_error:.6e}")

# Values of lambda
lambdas = [1e-4, 1e-2, 1e-1, 1, 10, 100]

# Storage for coefficient vectors
ridge_coefficients = {}
sparse_coefficients = {}

# Tolerance for determining whether a coefficient is zero
tol = 1e-6

# Ridge polynomial
for lam in lambdas:
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(Phi @ x - y) + 0.5 * lam * cp.sum_squares(x))
    problem = cp.Problem(objective)
    problem.solve()

    if x.value is None:
        print(f"\nRidge, lambda = {lam}: solver failed")
        continue

    ridge_coefficients[lam] = x.value.copy()
    residual = np.linalg.norm(Phi @ x.value - y)
    l2_norm = np.linalg.norm(x.value)
    l1_norm = np.linalg.norm(x.value, 1)
    sparsity = np.sum(np.abs(x.value) > tol)

    print(f"\nRidge, lambda = {lam}")
    print(f"Status       : {problem.status}")
    print(f"Residual     : {residual:.6e}")
    print(f"||x||_1      : {l1_norm:.6e}")
    print(f"||x||_2      : {l2_norm:.6e}")
    print(f"||x||_0      : {sparsity}")
    print("Coefficient vector:")
    print(x.value)

# Sparse polynomial
for lam in lambdas:
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(Phi @ x - y) + lam * cp.norm1(x))
    problem = cp.Problem(objective)
    problem.solve()

    if x.value is None:
        print(f"\nSparse, lambda = {lam}: solver failed")
        continue

    sparse_coefficients[lam] = x.value.copy()
    residual = np.linalg.norm(Phi @ x.value - y)
    l2_norm = np.linalg.norm(x.value)
    l1_norm = np.linalg.norm(x.value, 1)
    sparsity = np.sum(np.abs(x.value) > tol)

    print(f"\nSparse, lambda = {lam}")
    print(f"Status       : {problem.status}")
    print(f"Residual     : {residual:.6e}")
    print(f"||x||_1      : {l1_norm:.6e}")
    print(f"||x||_2      : {l2_norm:.6e}")
    print(f"||x||_0      : {sparsity}")
    print("Coefficient vector:")
    print(x.value)

# Summary table
print("\nSUMMARY")
print(f"{'Lambda':>10} {'Model':>10} {'Residual':>15} {'||x||_1':>15} {'||x||_2':>15} {'||x||_0':>10}")
print("-" * 80)

for lam in lambdas:
    if lam in ridge_coefficients:
        x = ridge_coefficients[lam]
        residual = np.linalg.norm(Phi @ x - y)
        l1_norm = np.linalg.norm(x, 1)
        l2_norm = np.linalg.norm(x)
        sparsity = np.sum(np.abs(x) > tol)
        print(f"{lam:10.1e} {'Ridge':>10} {residual:15.6e} {l1_norm:15.6e} {l2_norm:15.6e} {sparsity:10d}")

    if lam in sparse_coefficients:
        x = sparse_coefficients[lam]
        residual = np.linalg.norm(Phi @ x - y)
        l1_norm = np.linalg.norm(x, 1)
        l2_norm = np.linalg.norm(x)
        sparsity = np.sum(np.abs(x) > tol)
        print(f"{lam:10.1e} {'Sparse':>10} {residual:15.6e} {l1_norm:15.6e} {l2_norm:15.6e} {sparsity:10d}")