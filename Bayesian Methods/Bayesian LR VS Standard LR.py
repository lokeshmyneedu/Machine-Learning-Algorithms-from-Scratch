import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PREPARE REAL-WORLD DATA (Housing Prices)
# ==========================================
# X: Size in 1000s of sq ft
# y: Price in $100,000s
X_raw = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
y_raw = np.array([2.1, 2.4, 2.6, 2.8, 3.1, 3.2, 3.5, 3.8, 4.0, 4.2])

# Add a "Bias" column (intercept) of 1s to X
# This changes X from shape (10,) to (10, 2)
X = np.vstack([np.ones(len(X_raw)), X_raw]).T
y = y_raw

print(f"Data Loaded: {len(X)} houses.")

# ==========================================
# 2. STANDARD LINEAR REGRESSION (Frequentist)
# ==========================================
# Formula: w = (X^T * X)^-1 * X^T * y
# This finds the single "Best" line that minimizes error.
def standard_linear_regression(X, y):
    # Step 1: Compute X Transpose
    XT = X.T
    # Step 2: Compute (X^T * X)
    XTX = XT @ X
    # Step 3: Invert it
    XTX_inv = np.linalg.inv(XTX)
    # Step 4: Multiply by X^T * y
    w = XTX_inv @ (XT @ y)
    return w

w_standard = standard_linear_regression(X, y)
print(f"\n--- Standard Linear Regression ---")
print(f"Predicted Formula: Price = {w_standard[0]:.2f} + {w_standard[1]:.2f} * Size")


# ==========================================
# 3. BAYESIAN LINEAR REGRESSION
# ==========================================
# We don't just calculate weights 'w'; we calculate a DISTRIBUTION for 'w'.
# We need:
#   alpha: Precision (confidence) of our Prior belief
#   beta:  Precision of the data noise (how reliable is the data?)
def bayesian_linear_regression(X, y, alpha=1.0, beta=25.0):
    n_features = X.shape[1]
    
    # --- Step A: Define Prior ---
    # We assume weights are normally distributed around 0
    # Prior Precision Matrix (S_0_inv)
    S0_inv = alpha * np.eye(n_features)
    
    # --- Step B: Calculate Posterior (The "Training") ---
    # Formula: S_N_inv = S_0_inv + beta * (X^T * X)
    # This combines our Prior (S0) with the Data Evidence (XTX)
    SN_inv = S0_inv + beta * (X.T @ X)
    SN = np.linalg.inv(SN_inv)  # Covariance Matrix (Uncertainty of weights)
    
    # Formula: m_N = SN * (beta * X^T * y)
    # This calculates the Mean weights (Expected value)
    mN = SN @ (beta * (X.T @ y))
    
    return mN, SN

# Train the Bayesian Model
w_mean_bayes, w_cov_bayes = bayesian_linear_regression(X, y)

print(f"\n--- Bayesian Linear Regression ---")
print(f"Mean Weights: Intercept={w_mean_bayes[0]:.2f}, Slope={w_mean_bayes[1]:.2f}")
print("Uncertainty (Covariance) of Weights:\n", w_cov_bayes)


# ==========================================
# 4. PREDICTION & COMPARISON
# ==========================================
# Let's predict the price for a HUGE house (3,000 sq ft -> input 3.0)
# This is outside our training data, so uncertainty should be high.
new_house = np.array([1.0, 3.0]) 

# Standard Prediction (Single number)
price_std = new_house @ w_standard

# Bayesian Prediction (Distribution)
# Mean = x * w_mean
price_bayes_mean = new_house @ w_mean_bayes
# Variance = 1/beta + x^T * Covariance * x
# (1/beta is the inherent noise, the rest is model uncertainty)
price_bayes_var = (1/25.0) + (new_house @ w_cov_bayes @ new_house.T)
price_bayes_std = np.sqrt(price_bayes_var)

print(f"\n--- Prediction for 3,000 sq ft House ---")
print(f"Standard Model says: ${price_std:.2f}00,000")
print(f"Bayesian Model says: ${price_bayes_mean:.2f}00,000")
print(f"Bayesian Confidence: ${price_bayes_mean:.2f} ± ${price_bayes_std:.2f} (1 std dev)")
print(f"                     (Range: ${price_bayes_mean - price_bayes_std:.2f} to ${price_bayes_mean + price_bayes_std:.2f})")