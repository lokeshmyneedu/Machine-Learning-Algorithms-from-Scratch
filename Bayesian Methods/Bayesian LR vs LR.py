import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge

# ==========================================
# 1. SETUP: Generate Synthetic Data
# ==========================================
np.random.seed(42)

# True parameters
true_slope = 1.5
true_intercept = -2.0
noise_scale = 1.0

# Generate 10 data points (Small data makes Bayesian powerful)
N = 10
X = np.random.uniform(0, 5, size=N).reshape(-1, 1)  # Inputs (0 to 5)
y = true_slope * X.flatten() + true_intercept + np.random.normal(0, noise_scale, size=N)

# Generate test points for plotting the lines
X_test = np.linspace(-1, 6, 100).reshape(-1, 1)


# ==========================================
# 2. MODEL A: Ordinary Least Squares (OLS)
# ==========================================
ols = LinearRegression()
ols.fit(X, y)
y_ols_pred = ols.predict(X_test)


# ==========================================
# 3. MODEL B: Bayesian Regression (From Scratch)
# ==========================================
# Goal: Calculate Posterior Mean (w_map) and Covariance (S_N)
# Formula: w_map = S_N * (S_0^-1 * m_0 + beta * Phi^T * t)
# ------------------------------------------

# Hyperparameters (Assumptions)
beta = 1 / (noise_scale ** 2)  # Precision of the noise (known for this demo)
alpha = 1.0                    # Precision of the prior (assumption: weights are small)

# Add intercept column to X (design matrix Phi)
Phi = np.hstack([np.ones((N, 1)), X])
Phi_test = np.hstack([np.ones((len(X_test), 1)), X_test])

# Prior: Mean m0 = 0, Covariance S0 = (1/alpha) * I
m0 = np.zeros(2)
S0_inv = alpha * np.eye(2)

# Posterior Covariance Matrix: S_N = (S0^-1 + beta * Phi^T * Phi)^-1
SN_inv = S0_inv + beta * (Phi.T @ Phi)
SN = np.linalg.inv(SN_inv)

# Posterior Mean Weights: w_map = SN * (S0^-1 * m0 + beta * Phi^T * y)
# Note: Since m0 is 0, the first term vanishes
w_map = SN @ (beta * Phi.T @ y)

# Make Predictions
y_bayes_scratch_mean = Phi_test @ w_map

# Calculate Uncertainty (Standard Deviation of predictions)
# Variance = 1/beta + phi^T * SN * phi
y_bayes_scratch_var = []
for i in range(len(Phi_test)):
    phi_x = Phi_test[i]
    var_x = (1/beta) + phi_x.T @ SN @ phi_x
    y_bayes_scratch_var.append(var_x)
y_bayes_scratch_std = np.sqrt(y_bayes_scratch_var)


# ==========================================
# 4. MODEL C: Scikit-Learn BayesianRidge
# ==========================================
# This automatically tunes alpha and beta, so it may differ slightly from scratch
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)
y_sklearn_mean, y_sklearn_std = clf.predict(X_test, return_std=True)


# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 6))

# Plot Data
plt.scatter(X, y, color='black', label='Observed Data', zorder=10)
plt.plot(X_test, true_slope * X_test + true_intercept, 'k--', alpha=0.5, label='True Law (Unknown)')

# Plot OLS
plt.plot(X_test, y_ols_pred, color='red', linewidth=2, label=f'OLS Prediction (Slope={ols.coef_[0]:.2f})')

# Plot Bayesian (Scratch)
plt.plot(X_test, y_bayes_scratch_mean, color='blue', linewidth=2, label=f'Bayes Scratch (Slope={w_map[1]:.2f})')
plt.fill_between(X_test.flatten(), 
                 y_bayes_scratch_mean - 1.96 * y_bayes_scratch_std, 
                 y_bayes_scratch_mean + 1.96 * y_bayes_scratch_std, 
                 color='blue', alpha=0.1, label='Bayes Scratch Uncertainty (95%)')

# Plot Bayesian (Sklearn)
plt.plot(X_test, y_sklearn_mean, color='green', linestyle=':', linewidth=3, label=f'Sklearn Bayes (Slope={clf.coef_[0]:.2f})')

plt.title("Comparison: OLS vs Bayesian (Scratch) vs Scikit-Learn")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print Weights Comparison
print(f"{'Method':<20} | {'Intercept':<10} | {'Slope':<10}")
print("-" * 45)
print(f"{'True Values':<20} | {true_intercept:<10.2f} | {true_slope:<10.2f}")
print(f"{'OLS':<20} | {ols.intercept_:<10.2f} | {ols.coef_[0]:<10.2f}")
print(f"{'Bayes (Scratch)':<20} | {w_map[0]:<10.2f} | {w_map[1]:<10.2f}")
print(f"{'Bayes (Sklearn)':<20} | {clf.intercept_:<10.2f} | {clf.coef_[0]:<10.2f}")