import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel

# --- 1. Load Real World Data (Mauna Loa CO2) ---
print("Loading data... (this might take a few seconds)")
co2 = fetch_openml(data_id=41187, as_frame=True, parser='auto')
df = co2.frame

# Preprocessing: Create a float "Year" column (e.g., 1958.25)
df['date_float'] = df['year'] + (df['month'] - 1) / 12
df = df.sort_values('date_float')

# Take a subset (first 10 years) to make the visualization clear
df_subset = df[df['year'] < 1970]

X = df_subset['date_float'].values.reshape(-1, 1)
y = df_subset['co2'].values

# Creates test points for plotting smooth lines
X_test = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

# --- 2. Bayesian Linear Regression (The "Rigid" Approach) ---
# It assumes y = w*x + b (A straight line)
blr = BayesianRidge(compute_score=True)
blr.fit(X, y)
y_blr_mean, y_blr_std = blr.predict(X_test, return_std=True)

# --- 3. Gaussian Process (The "Flexible" Approach) ---
# We use a kernel that combines:
# - RBF (for smoothness/trend)
# - ExpSineSquared (specifically for periodic/seasonal data)
# - WhiteKernel (to handle noise)
kernel = 1.0 * RBF(length_scale=50.0) + \
         ExpSineSquared(length_scale=1.0, periodicity=1.0) + \
         WhiteKernel(noise_level=1.0)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gp.fit(X, y)
y_gp_mean, y_gp_std = gp.predict(X_test, return_std=True)

# --- 4. Visualization ---
plt.figure(figsize=(14, 6))

# Plot Data
plt.scatter(X, y, c='black', s=10, label='Real Data (Mauna Loa CO2)', zorder=3)

# Plot Bayesian Linear Regression
plt.plot(X_test, y_blr_mean, color='red', linewidth=2, label='Bayesian Linear Regression')
plt.fill_between(X_test.ravel(), y_blr_mean - 1.96*y_blr_std, y_blr_mean + 1.96*y_blr_std, 
                 color='red', alpha=0.2, label='BLR Uncertainty')

# Plot Gaussian Process
plt.plot(X_test, y_gp_mean, color='blue', linewidth=2, label='Gaussian Process')
plt.fill_between(X_test.ravel(), y_gp_mean - 1.96*y_gp_std, y_gp_mean + 1.96*y_gp_std, 
                 color='blue', alpha=0.2, label='GP Uncertainty')

plt.title("Real World Comparison: Bayesian Linear Regression vs. Gaussian Process")
plt.xlabel("Year")
plt.ylabel("CO2 Concentration (ppm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()