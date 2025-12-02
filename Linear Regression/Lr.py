import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
class CustomLinearRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 regularization: Optional[str] = None, lambda_reg: float = 0.01):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.cost_history = []
    
    # Feature Scaling Z-score normalization
    def _feature_scaling(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size:Optional[int]=None):
        # Initialize Parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []

        # Scale Features
        X_scaled = self._feature_scaling(X)

        if batch_size is None:
            batch_size = n_samples

        for i in range(0,n_samples,batch_size):
            X_batch = X_scaled[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            current_batch_size = X_batch.shape[0]

            # Forward Pass : Compute Predictions
            y_predicted = np.dot(X_batch, self.weights) + self.bias

            # Compute Gradient
            errors = y_predicted - y_batch
            # Gradient Descent Loop
            dw = (1 / current_batch_size) * np.dot(X_batch.T, errors)
            db = (1 / current_batch_size) * np.sum(errors)

            # Apply Regularization
            if self.regularization == 'l2':
                dw += self.lambda_reg * self.weights
            elif self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights)

            # Update Parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._feature_scaling(X)
        return np.dot(X_scaled, self.weights) + self.bias


if __name__ == "__main__":
    # 1. Load Data
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Prepare Models
    results = []
    
    # --- Configuration ---
    epochs = 100
    lr = 0.01
    
    # We need an external scaler for Sklearn comparisons
    scaler = StandardScaler()
    X_train_sk = scaler.fit_transform(X_train)
    X_test_sk = scaler.transform(X_test)

    print(f"Training models for {epochs} epochs...")

    # --- A. Custom Batch GD ---
    start = time.time()
    model_batch = CustomLinearRegression(learning_rate=lr, n_iterations=epochs)
    model_batch.fit(X_train, y_train, batch_size=None) # None = All samples
    time_batch = time.time() - start
    results.append({"Name": "Custom Batch GD", "Model": model_batch, "Time": time_batch, "Is_Sklearn": False})

    # --- B. Custom Mini-Batch GD ---
    start = time.time()
    model_mini = CustomLinearRegression(learning_rate=lr, n_iterations=epochs)
    model_mini.fit(X_train, y_train, batch_size=32) # 32 samples at a time
    time_mini = time.time() - start
    results.append({"Name": "Custom Mini-Batch", "Model": model_mini, "Time": time_mini, "Is_Sklearn": False})

    # --- C. Custom SGD (Batch size 1) ---
    # Note: Pure Python SGD is very slow due to loops, we reduce epochs for demo or wait
    start = time.time()
    model_sgd = CustomLinearRegression(learning_rate=0.001, n_iterations=epochs) # Lower LR for SGD
    model_sgd.fit(X_train, y_train, batch_size=1) 
    time_sgd = time.time() - start
    results.append({"Name": "Custom SGD (BS=1)", "Model": model_sgd, "Time": time_sgd, "Is_Sklearn": False})

    # --- D. Sklearn SGDRegressor (Standard) ---
    start = time.time()
    sk_model = SGDRegressor(learning_rate='invscaling', eta0=lr, max_iter=epochs, tol=1e-3, random_state=42)
    sk_model.fit(X_train_sk, y_train)
    time_sk = time.time() - start
    results.append({"Name": "Sklearn SGD (Default)", "Model": sk_model, "Time": time_sk, "Is_Sklearn": True})

    # --- E. Sklearn Mini-Batch (Simulated via partial_fit) ---
    start = time.time()
    sk_mini = SGDRegressor(learning_rate='invscaling', eta0=0.001,power_t=0.25, random_state=42)
    batch_size = 32
    n_samples = X_train_sk.shape[0]
    
    # Manual epoch loop
    for _ in range(epochs):
        idx = np.random.permutation(n_samples)
        X_shuf, y_shuf = X_train_sk[idx], y_train[idx]
        
        for i in range(0, n_samples, batch_size):
            X_b = X_shuf[i:i+batch_size]
            y_b = y_shuf[i:i+batch_size]
            
            # partial_fit updates weights based on JUST this batch
            sk_mini.partial_fit(X_b, y_b)
            
    time_sk_mini = time.time() - start
    results.append({"Name": "Sklearn Mini-Batch", "Model": sk_mini, "Time": time_sk_mini, "Is_Sklearn": True})


    # ==========================================
    # 3. COMPARE RESULTS
    # ==========================================
    print("\n" + "="*95)
    print(f"{'Model Name':<25} | {'MSE':<10} | {'R2 Score':<10} | {'Time (sec)':<10} | {'Bias/Intercept':<10}")
    print("-" * 95)

    final_preds = {}

    for res in results:
        model = res["Model"]
        name = res["Name"]
        
        if res["Is_Sklearn"]:
            preds = model.predict(X_test_sk)
            intercept = model.intercept_[0]
        else:
            preds = model.predict(X_test)
            intercept = model.bias

        final_preds[name] = preds
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"{name:<25} | {mse:.4f}     | {r2:.4f}     | {res['Time']:.4f}     | {intercept:.4f}")

    print("="*95)

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:50], label='Actual', color='black', marker='o', linestyle='', alpha=0.6)

    styles = ['-', '--', '-.', ':', '-']
    for i, (name, preds) in enumerate(final_preds.items()):
        plt.plot(preds[:50], label=name, linestyle=styles[i%len(styles)], alpha=0.8)

    plt.title(f"Comparison of GD Variants (First 50 Samples)")
    plt.ylabel("House Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    # Load Real Data
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Custom Model
    custom_model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    
    # Train Sklearn Model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sklearn_model = SGDRegressor(
        learning_rate='invscaling', # Let sklearn lower the rate as it trains
        eta0=0.01,                 # Start at 0.01
        max_iter=1000,
        penalty='l2',
        alpha=0.01,
        random_state=42
    )
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_preds = sklearn_model.predict(X_test_scaled)

    # ==========================================
    # 5. COMPARE RESULTS
    # ==========================================
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Custom Model':<15} | {'Scikit-Learn (SGD)':<15}")
    print("-" * 60)
    print(f"{'MSE (Error)':<20} | {mean_squared_error(y_test, custom_preds):.4f}          | {mean_squared_error(y_test, sklearn_preds):.4f}")
    print(f"{'R2 Score':<20} | {r2_score(y_test, custom_preds):.4f}          | {r2_score(y_test, sklearn_preds):.4f}")
    # Handle intercept printing robustly (intercept_ may be array-like)
    try:
        sklearn_intercept = float(np.ravel(sklearn_model.intercept_)[0])
    except Exception:
        sklearn_intercept = float(sklearn_model.intercept_) if np.isscalar(sklearn_model.intercept_) else 0.0
    print(f"{'Bias (Intercept)':<20} | {custom_model.bias:.4f}          | {sklearn_intercept:.4f}")
    print("="*60)

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 5))

    # Plot a slice of 50 random points for clarity
    plt.plot(y_test[:50], label='Actual Values', color='black', marker='o', linestyle='')
    plt.plot(custom_preds[:50], label='Custom Preds', alpha=0.7)
    plt.plot(sklearn_preds[:50], label='Sklearn Preds', alpha=0.7, linestyle='--')

    plt.title("Actual vs Predicted (First 50 Test Samples)")
    plt.ylabel("Median House Value ($100k)")
    plt.legend()
    plt.show()