import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. PREPARE REAL WORLD DATA ---
# Load California Housing Data
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Standardize features (CRITICAL for Bayesian methods)
# If features have vastly different scales (e.g., Rooms vs Income), 
# the single "alpha" prior won't work well for all of them.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        """
        alpha: Precision (1/variance) of the Prior. (High alpha = strong belief weights are near 0)
        beta:  Precision (1/variance) of the Noise. (High beta = data is very accurate)
        """
        self.alpha = alpha
        self.beta = beta
        self.mean_posterior = None
        self.cov_posterior = None

    def fit(self, X, y):
        # 1. Add bias term (column of 1s) to X
        # Shape becomes (N_samples, N_features + 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        N, M = X_b.shape # N samples, M features (including bias)
        I = np.eye(M)    # Identity matrix of size M
        
        # 2. Calculate Posterior Covariance Matrix (S_N)
        # Formula: (alpha * I + beta * X.T @ X)^-1
        # We use np.linalg.inv for inversion
        hessian = (self.alpha * I) + (self.beta * X_b.T @ X_b)
        self.cov_posterior = np.linalg.inv(hessian)
        
        # 3. Calculate Posterior Mean Vector (m_N)
        # Formula: beta * S_N @ X.T @ y
        self.mean_posterior = self.beta * self.cov_posterior @ X_b.T @ y

    def predict(self, X):
        # 1. Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 2. Predict Mean (y_pred)
        # Just like standard regression: y = X * w
        y_pred = X_b @ self.mean_posterior
        
        # 3. Predict Variance (Uncertainty)
        # Formula: 1/beta + x.T @ S_N @ x
        # This gives us the "confidence tube" around the line
        y_var = []
        for x in X_b:
            # mathematical term: sigma^2 = 1/beta + x^T * S_N * x
            var_term = (1 / self.beta) + (x.T @ self.cov_posterior @ x)
            y_var.append(var_term)
            
        return y_pred, np.array(y_var)
# --- 2. TRAIN SCIKIT-LEARN MODEL ---
# Notice we don't pass alpha/beta. It learns them!
sklearn_model = BayesianRidge(compute_score=True)
sklearn_model.fit(X_train, y_train)

# --- 3. TRAIN OUR SCRATCH MODEL ---
# (Using the class we wrote in the previous turn)
# To make a fair comparison, we will "cheat" and plug in the 
# optimal alpha/lambda that sklearn found.
# Sklearn's 'lambda' is our 'alpha' (precision of weights)
# Sklearn's 'alpha' is our 'beta' (precision of noise)
optimal_alpha_from_sklearn = sklearn_model.lambda_
optimal_beta_from_sklearn = sklearn_model.alpha_

scratch_model = BayesianLinearRegression(
    alpha=optimal_alpha_from_sklearn, 
    beta=optimal_beta_from_sklearn
)
scratch_model.fit(X_train, y_train)

# --- 4. COMPARE RESULTS ---
print(f"{'METRIC':<20} | {'SKLEARN':<15} | {'SCRATCH (Ours)':<15}")
print("-" * 55)

# Compare Coefficients (Weights)
# We take the first 3 features just for brevity
for i in range(3):
    feat = feature_names[i]
    w_sklearn = sklearn_model.coef_[i]
    # Our scratch model includes bias in index 0, so we shift by +1 for features
    w_scratch = scratch_model.mean_posterior[i+1] 
    print(f"Coeff: {feat:<13} | {w_sklearn:.5f}         | {w_scratch:.5f}")

print("-" * 55)

# Compare Predictions on first 5 test points
pred_sklearn, std_sklearn = sklearn_model.predict(X_test[:5], return_std=True)
pred_scratch, var_scratch = scratch_model.predict(X_test[:5])
std_scratch = np.sqrt(var_scratch).flatten() # Convert variance to std deviation

print(f"Prediction (House 1) | {pred_sklearn[0]:.5f}         | {pred_scratch[0]:.5f}")
print(f"Uncertainty (StdDev) | {std_sklearn[0]:.5f}         | {std_scratch[0]:.5f}")

# --- 5. VISUALIZATION ---
# Let's plot the "Learned" parameters over iterations (Sklearn feature)
plt.figure(figsize=(10, 5))
plt.plot(sklearn_model.scores_)
plt.title("Sklearn Optimization: Maximizing Log-Marginal Likelihood")
plt.xlabel("Iterations")
plt.ylabel("Log-Marginal Likelihood")
plt.grid(True)
plt.show()