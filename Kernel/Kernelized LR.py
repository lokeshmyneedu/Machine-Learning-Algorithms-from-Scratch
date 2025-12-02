import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

# 1. Generate Synthetic Non-Linear Data (Sine Wave)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# Add some noise
y[::5] += 3 * (0.5 - np.random.rand(16))

# 2. Fit Standard Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# 3. Fit Kernel Ridge Regression (RBF Kernel)
# alpha = regularization strength, gamma = kernel width (how 'wiggly' the curve is)
kernel_model = KernelRidge(kernel='rbf', alpha=0.1, gamma=1.0)
kernel_model.fit(X, y)

# 4. Predict on a grid to plot smooth curves
X_plot = np.linspace(0, 5, 1000)[:, None]
y_linear_pred = linear_model.predict(X_plot)
y_kernel_pred = kernel_model.predict(X_plot)

# 5. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='gray', label='Training Data', s=30)
plt.plot(X_plot, y_linear_pred, c='red', linewidth=2, label='Linear Regression (Underfitting)')
plt.plot(X_plot, y_kernel_pred, c='blue', linewidth=2, label='Kernel Ridge Regression (RBF)')

plt.xlabel('Input Feature (x)')
plt.ylabel('Target (y)')
plt.title('Linear Regression vs. Kernelized Linear Regression')
plt.legend()
plt.show()