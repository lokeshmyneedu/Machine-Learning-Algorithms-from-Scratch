import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)
n_samples = 100

# Features: sqft, bedrooms, age, distance_to_city
X = np.column_stack([
    np.random.uniform(1000, 4000, n_samples),  # sqft
    np.random.randint(1, 6, n_samples),         # bedrooms
    np.random.uniform(0, 50, n_samples),        # age
    np.random.uniform(1, 30, n_samples)         # distance (km)
])

# True relationship (simple)
y = 100*X[:, 0] + 20000*X[:, 1] - 1000*X[:, 2] - 5000*X[:, 3] + np.random.normal(0, 10000, n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Feature ranges (training data):")
for i, name in enumerate(['Sqft', 'Beds', 'Age', 'Distance']):
    print(f"  {name}: {X_train[:, i].min():.1f} to {X_train[:, i].max():.1f}")

# No Scalinag and regularization
class SimpleLinearRegression:
    def fit(self, X, y, lr=0.0000001, iterations=1000):
        X_b = np.c_[np.ones(len(X)), X]  # Add bias
        self.weights = np.zeros(X_b.shape[1])
        
        for _ in range(iterations):
            gradient = (2/len(y)) * X_b.T @ (X_b @ self.weights - y)
            self.weights -= lr * gradient
        
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return X_b @ self.weights

model1 = SimpleLinearRegression()
model1.fit(X_train, y_train)
print("Weights:", model1.weights)
# Tiny learning rate needed, slow convergence, weights hard to interpret

# With Scaling and No Regularization
class FeatureScaler:
    def fit_transform(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return (X - self.mean) / (self.std + 1e-8)
    
    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

scaler = FeatureScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled feature ranges:")
for i, name in enumerate(['Sqft', 'Beds', 'Age', 'Distance']):
    print(f"  {name}: {X_train_scaled[:, i].min():.2f} to {X_train_scaled[:, i].max():.2f}")

model2 = SimpleLinearRegression()
model2.fit(X_train_scaled, y_train, lr=0.1, iterations=1000)  # Much larger lr works!
print("\nWeights (scaled features):", model2.weights)

# With Scaling and L2 Regularization
# Add 10 irrelevant noise features
X_train_noisy = np.c_[X_train_scaled, np.random.randn(len(X_train_scaled), 10)]
X_test_noisy = np.c_[X_test_scaled, np.random.randn(len(X_test_scaled), 10)]

class RidgeRegression:
    def fit(self, X, y, lr=0.1, iterations=1000, lambda_reg=0.0):
        X_b = np.c_[np.ones(len(X)), X]
        self.weights = np.zeros(X_b.shape[1])
        
        for _ in range(iterations):
            gradient = (2/len(y)) * X_b.T @ (X_b @ self.weights - y)
            
            # Add regularization gradient (skip bias at index 0)
            reg_gradient = (2 * lambda_reg / len(y)) * self.weights
            reg_gradient[0] = 0
            
            self.weights -= lr * (gradient + reg_gradient)
        
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones(len(X)), X]
        return X_b @ self.weights

# Without regularization
model_no_reg = RidgeRegression()
model_no_reg.fit(X_train_noisy, y_train, lambda_reg=0.0)

# With regularization
model_with_reg = RidgeRegression()
model_with_reg.fit(X_train_noisy, y_train, lambda_reg=10.0)

print("Weights WITHOUT regularization:")
print("  Real features:", model_no_reg.weights[1:5])
print("  Noise features:", model_no_reg.weights[5:])

print("\nWeights WITH regularization:")
print("  Real features:", model_with_reg.weights[1:5])
print("  Noise features:", model_with_reg.weights[5:])

