import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel

# --- 1. Load Data ---
co2 = fetch_openml(data_id=41187, as_frame=True, parser='auto')
df = co2.frame
df['date_float'] = df['year'] + (df['month'] - 1) / 12
df = df.sort_values('date_float')

# Train on data before 1970
X_train = df[df['year'] < 1970]['date_float'].values.reshape(-1, 1)
y_train = df[df['year'] < 1970]['co2'].values

# Predict way into the future (up to 1990) to see the difference
X_test = np.linspace(X_train.min(), 1990, 500).reshape(-1, 1)

# --- 2. The Additive Model (Trend + Seasonality) ---
# Logic: The seasonal wave is constant. It will wobble exactly the same way forever.
k_add = 50.0**2 * RBF(length_scale=50.0) + \
        2.0**2 * ExpSineSquared(length_scale=1.0, periodicity=1.0) + \
        WhiteKernel(noise_level=1.0)

gp_add = GaussianProcessRegressor(kernel=k_add, normalize_y=True)
gp_add.fit(X_train, y_train)
y_add, std_add = gp_add.predict(X_test, return_std=True)

# --- 3. The Multiplicative Model (Locally Periodic) ---
# Logic: We MULTIPLY RBF * Periodic. 
# This means the periodicity only exists where the RBF is strong.
# As we move away from the data, the RBF drops to 0, so the wave should die out.
k_mult = 50.0**2 * RBF(length_scale=50.0) * \
         ExpSineSquared(length_scale=1.0, periodicity=1.0) + \
         WhiteKernel(noise_level=1.0)

gp_mult = GaussianProcessRegressor(kernel=k_mult, normalize_y=True)
gp_mult.fit(X_train, y_train)
y_mult, std_mult = gp_mult.predict(X_test, return_std=True)

# --- 4. Visualization ---
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Addition
axes[0].scatter(X_train, y_train, c='black', s=10, label='Training Data')
axes[0].plot(X_test, y_add, color='blue', label='Prediction')
axes[0].fill_between(X_test.ravel(), y_add - 1.96*std_add, y_add + 1.96*std_add, color='blue', alpha=0.2)
axes[0].set_title("ADDITION Kernel (RBF + Periodic)\nNote: The wave continues with constant strength forever.")
axes[0].legend()

# Plot Multiplication
axes[1].scatter(X_train, y_train, c='black', s=10, label='Training Data')
axes[1].plot(X_test, y_mult, color='red', label='Prediction')
axes[1].fill_between(X_test.ravel(), y_mult - 1.96*std_mult, y_mult + 1.96*std_mult, color='red', alpha=0.2)
axes[1].set_title("MULTIPLICATION Kernel (RBF * Periodic)\nNote: The wave 'decays' as we get further from known data.")
axes[1].legend()

plt.show()