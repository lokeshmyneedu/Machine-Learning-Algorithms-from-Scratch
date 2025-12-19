import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Create a Sine Wave with Noise
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0]) # y = sin(x) + noise

# 2. Train Random Forest (Parallel)
# Uses deep trees (max_depth=10) to try and fit the data
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X, y)

# 3. Train Gradient Boosting (Sequential)
# Uses SHALLOW trees (max_depth=3) but learns from mistakes
# learning_rate=0.1 means we take 10% of the new tree's advice each time
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gbr.fit(X, y)

# 4. Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_rf = rf.predict(X_test)
y_gbr = gbr.predict(X_test)

# 5. Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=20, edgecolor="black", c="gray", label="Data")
plt.plot(X_test, y_rf, color="blue", label="Random Forest (Bagging)", linewidth=2, linestyle="--")
plt.plot(X_test, y_gbr, color="red", label="Gradient Boosting (Boosting)", linewidth=2)
plt.title("Random Forest vs. Gradient Boosting")
plt.legend()
plt.show()