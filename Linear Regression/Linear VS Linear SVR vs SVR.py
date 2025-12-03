import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 1. Generate "Real World" Data (Sine wave = Seasonal Sales)
# X = Temperature, y = Sales
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# Add noise (measurement errors)
y[::5] += 3 * (0.5 - np.random.rand(16))

# 2. Fit Models
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Linear SVR
# C=1.0 is regularization, epsilon=0.1 is the "tube" width
svr_lin = SVR(kernel='linear', C=100, epsilon=0.1) 
svr_lin.fit(X, y)

# Complex SVR (RBF Kernel)
# Gamma defines how "curvy" the model is allowed to be
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X, y)

# 3. Predict for visualization
X_plot = np.linspace(0, 5, 100)[:, None]
y_lin = lin_reg.predict(X_plot)
y_svr_lin = svr_lin.predict(X_plot)
y_svr_rbf = svr_rbf.predict(X_plot)

# 4. Visualization
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='darkorange', label='Data (Sales vs Temp)', s=30, alpha=0.6)
plt.plot(X_plot, y_lin, color='navy', lw=2, label='Linear Regression')
plt.plot(X_plot, y_svr_lin, color='green', lw=2, linestyle='--', label='Linear SVR')
plt.plot(X_plot, y_svr_rbf, color='red', lw=2, label='Complex SVR (RBF)')

plt.xlabel('Temperature (Normalized)')
plt.ylabel('Sales Volume')
plt.title('Regression: Linear vs SVR (Linear) vs SVR (RBF)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()