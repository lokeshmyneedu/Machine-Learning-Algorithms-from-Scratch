import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Simulate "Real World" Data (Car Physics)
# ==========================================
# Relationship: MPG drops as HP increases, but not linearly (Inverse/Curved)
np.random.seed(42)
n_samples = 200

# Horsepower (feature)
X = np.sort(np.random.uniform(50, 350, n_samples)).reshape(-1, 1)

# MPG (target) - Underlying physics with noise
y = 4000 / X.ravel() + np.random.normal(0, 2, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Build the Model Pipeline
# ==========================================
# Polynomial features can create huge values (e.g., 300^3 = 27,000,000).
# Gradient Descent (SGD) will FAIL if data is not scaled.
# We must scale AFTER generating polynomial features.

pipeline = Pipeline([
    # Step A: Create non-linear features (x, x^2, x^3)
    ('poly', PolynomialFeatures(degree=4, include_bias=False)),
    
    # Step B: Scale features (Crucial for SGD!)
    ('scaler', StandardScaler()),
    
    # Step C: The Regressor with Optimization and Regularization
    ('sgd', SGDRegressor(
        loss='squared_error',   # The standard error metric
        penalty='elasticnet',   # REGULARIZATION: Mix of L1 and L2
        alpha=0.1,              # Regularization strength
        l1_ratio=0.5,           # 50% Lasso, 50% Ridge
        max_iter=5000,          # OPTIMIZER: Run for many iterations
        learning_rate='adaptive', # OPTIMIZER: Adjust step size automatically
        eta0=0.01,              # Initial learning rate
        random_state=42
    ))
])

# ==========================================
# 3. Train and Evaluate
# ==========================================
print("Training model with SGD...")
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Polynomial Degree: 4")
print(f"R-squared Score: {r2:.4f} (Closer to 1.0 is better)")
print(f"RMSE: {np.sqrt(mse):.4f} MPG")

# ==========================================
# 4. Visualization
# ==========================================
# Create a smooth line for plotting the model's learned curve
X_plot = np.linspace(50, 350, 200).reshape(-1, 1)
y_plot = pipeline.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='black', alpha=0.6, label='Test Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='SGD Polynomial Fit')
plt.title(f'Polynomial Regression with SGD & ElasticNet\n(R^2: {r2:.2f})')
plt.xlabel('Horsepower')
plt.ylabel('Miles Per Gallon (MPG)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()