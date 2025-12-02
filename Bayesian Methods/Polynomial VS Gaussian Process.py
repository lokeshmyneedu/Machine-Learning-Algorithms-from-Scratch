import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import make_pipeline

# --- 1. Generate Data (Sine wave with a gap) ---
np.random.seed(1)
X = np.sort(np.random.rand(20, 1) * 10, axis=0)
# Remove data in the middle (between 3 and 7) to create a "gap"
mask = (X < 3) | (X > 7)
X_train = X[mask].reshape(-1, 1)
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])

# Test points (to visualize the curve)
X_test = np.linspace(0, 10, 500).reshape(-1, 1)

# --- 2. High-Degree Polynomial Regression (The "Danger") ---
# Degree 15 is excessive, but it illustrates Runge's Phenomenon
degree = 15
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train, y_train)
y_poly = poly_model.predict(X_test)

# --- 3. Gaussian Process (The "Safe" Approach) ---
kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(X_train, y_train)
y_gp, std_gp = gp.predict(X_test, return_std=True)

# --- 4. Visualization ---
plt.figure(figsize=(12, 6))

# Plot True Function (for reference)
plt.plot(X_test, np.sin(X_test), 'k--', alpha=0.5, label='True Sine Wave')

# Plot Training Data
plt.scatter(X_train, y_train, c='black', s=50, zorder=10, label='Training Data')

# Plot Polynomial Regression
plt.plot(X_test, y_poly, color='red', linewidth=2, label=f'Polynomial (Degree {degree})')

# Plot Gaussian Process
plt.plot(X_test, y_gp, color='blue', linewidth=2, label='Gaussian Process')
plt.fill_between(X_test.ravel(), y_gp - 1.96*std_gp, y_gp + 1.96*std_gp, 
                 color='blue', alpha=0.1, label='GP Uncertainty')

plt.ylim(-2.5, 2.5)  # Limit y-axis to ignore the worst polynomial explosions
plt.title("Runge's Phenomenon: Polynomial vs. Gaussian Process")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()