import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SklearnLR

# ==========================================
# 1. DATA LOADING, SELECTION & SCALING
# ==========================================
def prepare_data():
    print("--- [1] Loading and Preprocessing Data ---")
    # Real-world dataset: Breast Cancer Wisconsin (Diagnostic)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"Original shape: {X.shape}")

    # Feature Selection: Select top 10 features using ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    # Get boolean mask of selected features
    mask = selector.get_support()
    # Get list of original feature names
    feature_names = data.feature_names
    # Filter
    selected_features = feature_names[mask]
    
    print(f"Shape after Feature Selection: {X_selected.shape}")
    print("The Top 10 Features are:", selected_features)
    print(f"Shape after Feature Selection: {X_selected.shape}")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Feature Scaling: CRITICAL for Gradient Descent convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# ==========================================
# 2. LOGISTIC REGRESSION FROM SCRATCH
# ==========================================
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, iterations=1000, method='gd'):
        self.lr = learning_rate
        self.iterations = iterations
        self.method = method # 'gd' (Gradient Descent) or 'newton' (Newton's Method)
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        # Clip z for numerical stability to prevent overflow/underflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # 1. Linear Model: z = w.x + b
            linear_model = np.dot(X, self.weights) + self.bias
            # 2. Prediction (Hypothesis)
            y_predicted = self.sigmoid(linear_model)

            # 3. Compute Gradients (Partial Derivatives)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. OPTIMIZATION
            if self.method == 'gd':
                # --- Gradient Descent Update ---
                # w = w - learning_rate * gradient
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            elif self.method == 'newton':
                # --- Newton's Method Update ---
                # w = w - Hessian^-1 * gradient
                # Hessian Calculation: H = (1/m) * X.T * S * X
                # S is diagonal matrix where S_ii = pred * (1 - pred)
                
                S = np.diag(y_predicted * (1 - y_predicted))
                H = (1 / n_samples) * np.dot(np.dot(X.T, S), X)
                
                # Regularization (Ridge) to ensure H is invertible
                lambda_reg = 1e-5
                H += lambda_reg * np.eye(n_features)
                
                # Update weights using Pseudo-Inverse of Hessian
                try:
                    H_inv = np.linalg.inv(H)
                    update = np.dot(H_inv, dw)
                    self.weights -= update
                    # For simplicity in Newton's, we'll use simple GD for bias or treat bias as feature column
                    # Here we just update bias with GD to keep the matrix math simple for this example
                    self.bias -= self.lr * db 
                except np.linalg.LinAlgError:
                    print("Hessian singular, switching to GD step")
                    self.weights -= self.lr * dw

            # Track Loss (Binary Cross Entropy)
            loss = -np.mean(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9))
            self.loss_history.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

# ==========================================
# 3. TENSORFLOW / KERAS IMPLEMENTATION
# ==========================================
def train_tensorflow(X_train, y_train, X_test):
    print("\n--- Training TensorFlow/Keras Model ---")
    model = tf.keras.Sequential([
        # Single Dense layer with 1 unit and sigmoid is exactly Logistic Regression
        tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],), activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    preds = model.predict(X_test)
    return [1 if p > 0.5 else 0 for p in preds]

# ==========================================
# 4. PYTORCH IMPLEMENTATION
# ==========================================
class PyTorchLR(torch.nn.Module):
    def __init__(self, n_features):
        super(PyTorchLR, self).__init__()
        # Linear layer: input_dim -> 1 output
        self.linear = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def train_pytorch(X_train, y_train, X_test):
    print("\n--- Training PyTorch Model ---")
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = PyTorchLR(X_train.shape[1])
    criterion = torch.nn.BCELoss() # Binary Cross Entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training Loop
    for epoch in range(100):
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Predict
    with torch.no_grad():
        y_predicted = model(X_test_t)
        y_cls = y_predicted.round()
        return y_cls.numpy().flatten()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Prepare Data
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. From Scratch (Gradient Descent)
    print("\n--- Training Scratch (Gradient Descent) ---")
    model_gd = LogisticRegressionScratch(learning_rate=0.01, iterations=1000, method='gd')
    model_gd.fit(X_train, y_train)
    preds_gd = model_gd.predict(X_test)
    acc_gd = accuracy_score(y_test, preds_gd)

    # 3. From Scratch (Newton's Method)
    print("--- Training Scratch (Newton's Method) ---")
    model_newton = LogisticRegressionScratch(learning_rate=1.0, iterations=50, method='newton')
    model_newton.fit(X_train, y_train)
    preds_newton = model_newton.predict(X_test)
    acc_newton = accuracy_score(y_test, preds_newton)

    # 4. Scikit-Learn (The Benchmark)
    print("\n--- Training Scikit-Learn ---")
    model_sk = SklearnLR()
    model_sk.fit(X_train, y_train)
    preds_sk = model_sk.predict(X_test)
    acc_sk = accuracy_score(y_test, preds_sk)

    # 5. TensorFlow
    preds_tf = train_tensorflow(X_train, y_train, X_test)
    acc_tf = accuracy_score(y_test, preds_tf)

    # 6. PyTorch
    preds_torch = train_pytorch(X_train, y_train, X_test)
    acc_torch = accuracy_score(y_test, preds_torch)

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n" + "="*40)
    print("FINAL RESULTS (Accuracy)")
    print("="*40)
    print(f"Scratch (Gradient Descent): {acc_gd:.4f}")
    print(f"Scratch (Newton's Method):  {acc_newton:.4f}")
    print(f"Scikit-Learn:               {acc_sk:.4f}")
    print(f"TensorFlow:                 {acc_tf:.4f}")
    print(f"PyTorch:                    {acc_torch:.4f}")
    print("="*40)