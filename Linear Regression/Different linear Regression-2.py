import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Real-World Data (California Housing)
# Target: Median House Value (in $100k)
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(f"Dataset Shape: {X.shape}")
print("-" * 50)

# 2. Preprocessing
# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (CRITICAL for Gradient Descent and Regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results for comparison
results = {}

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"[{name}]")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2 Score: {r2:.4f}\n")

# --- METHOD A: Normal Equation (Manual Linear Algebra) ---
# Formula: theta = (X^T * X)^-1 * X^T * y
# We must add a bias column (x0 = 1) manually for the matrix math
print("--- A. Normal Equation ---")
X_b_train = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
X_b_test = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]

# Calculate Theta
theta_best = np.linalg.inv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)

# Make predictions
y_pred_norm = X_b_test.dot(theta_best)
evaluate_model("Normal Equation", y_test, y_pred_norm)


# --- METHOD B: Ordinary Least Squares (OLS) ---
# The standard scikit-learn approach
print("--- B. Ordinary Least Squares (sklearn) ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_ols = lin_reg.predict(X_test_scaled)
evaluate_model("OLS (LinearRegression)", y_test, y_pred_ols)


# --- METHOD C: Gradient Descent (SGD) ---
# Stochastic Gradient Descent
print("--- C. Gradient Descent (SGD) ---")
# max_iter=1000, tol=1e-3 are standard stopping criteria
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.01, random_state=42)
sgd_reg.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_reg.predict(X_test_scaled)
evaluate_model("SGD (Gradient Descent)", y_test, y_pred_sgd)


# --- METHOD D: Regularization (Ridge, Lasso, ElasticNet) ---
print("--- D. Regularization ---")

# 1. Ridge (L2 Regularization)
# Alpha is the regularization strength
ridge_reg = Ridge(alpha=1.0, random_state=42)
ridge_reg.fit(X_train_scaled, y_train)
evaluate_model("Ridge (L2)", y_test, ridge_reg.predict(X_test_scaled))

# 2. Lasso (L1 Regularization)
# Useful for feature selection (can shrink coefs to 0)
lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X_train_scaled, y_train)
evaluate_model("Lasso (L1)", y_test, lasso_reg.predict(X_test_scaled))

# 3. Elastic Net (L1 + L2)
# l1_ratio=0.5 means 50% L1 and 50% L2
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)
evaluate_model("Elastic Net", y_test, elastic_net.predict(X_test_scaled))

# --- Summary Comparison ---
print("-" * 50)
print("SUMMARY COMPARISON DATAFRAME")
df_results = pd.DataFrame(results).T
print(df_results.sort_values(by="MSE"))