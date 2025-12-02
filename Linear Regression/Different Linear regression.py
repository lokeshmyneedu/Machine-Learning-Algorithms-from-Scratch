import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error
import time

# 1. Load Data
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scaling (Essential for solvers like SGD and Conjugate Gradient)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = []

def benchmark_model(name, model):
    start = time.time()
    model.fit(X_train_scaled, y_train)
    duration = time.time() - start
    mse = mean_squared_error(y_test, model.predict(X_test_scaled))
    results.append({'Method': name, 'MSE': mse, 'Time (s)': duration})

print(f"--- Benchmarking Solvers on California Housing Dataset ---")

# --- 1. SVD (Singular Value Decomposition) ---
# Standard LinearRegression in sklearn uses SVD (via scipy.linalg.lstsq)
# It solves X = U Sigma V^T
benchmark_model("OLS (SVD based)", LinearRegression())

# --- 2. Cholesky Decomposition ---
# Used for Ridge Regression. Very fast, closed-form solution.
# Solves (X^T X + alpha I) theta = X^T y using Cholesky factorization.
benchmark_model("Ridge (Cholesky)", Ridge(alpha=1.0, solver='cholesky'))

# --- 3. Conjugate Gradient (Sparse CG) ---
# Iterative method. Excellent for large datasets where X^T X is too expensive to compute.
benchmark_model("Ridge (Conjugate Gradient)", Ridge(alpha=1.0, solver='sparse_cg'))

# --- 4. Stochastic Gradient Descent (SGD) ---
# You mentioned SGDRegressor - I used it in the previous example, 
# but here it is again for comparison.
benchmark_model("SGD Regressor", SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))

# --- 5. Coordinate Descent ---
# This is the default solver for LASSO.
# It minimizes the function along one coordinate (feature) at a time, cycling through them.
# It works well with the L1 penalty (which has sharp corners).
benchmark_model("Lasso (Coordinate Descent)", Lasso(alpha=0.1))

# --- 6. SAG (Stochastic Average Gradient) ---
# A variation of Gradient Descent often used in Ridge.
benchmark_model("Ridge (SAG)", Ridge(alpha=1.0, solver='sag'))

# --- Comparison ---
df = pd.DataFrame(results)
print(df.sort_values(by="MSE"))