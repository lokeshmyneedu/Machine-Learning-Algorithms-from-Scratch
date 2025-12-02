import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
import tensorflow as tf

# ==========================================
# 1. PREPARE REAL-WORLD DATA
# ==========================================
def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Convert labels to {-1, 1} for Hinge Loss formulation
    # Original is {0, 1}, we map 0 -> -1
    y = np.where(y == 0, -1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling is CRITICAL for SVM convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()
results = []

# ==========================================
# 2. SVM FROM SCRATCH (Primal Form + GD)
# ==========================================
class SVM_Scratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if point satisfies the margin condition
                # Condition: y_i * (w.x + b) >= 1
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Point is correctly classified and outside margin
                    # Gradient of hinge loss is 0, only update regularization
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Point is misclassified or inside margin
                    # Gradient includes hinge loss term
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = -y[idx]

                # Update weights (Gradient Descent)
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        # Linear output: w.x + b
        linear_output = np.dot(X, self.w) + self.b
        # Sign function: -1 if < 0, +1 if >= 0
        return np.sign(linear_output)

# --- Train Scratch Model ---
start = time.time()
model_scratch = SVM_Scratch(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
model_scratch.fit(X_train, y_train)
scratch_time = time.time() - start
preds_scratch = model_scratch.predict(X_test)
acc_scratch = accuracy_score(y_test, preds_scratch)
results.append(["Scratch (NumPy)", acc_scratch, scratch_time])


# ==========================================
# 3. SCIKIT-LEARN IMPLEMENTATION
# ==========================================
# LinearSVC is faster than SVC(kernel='linear') for large datasets
start = time.time()
model_sklearn = LinearSVC(loss='hinge', max_iter=1000, C=1.0, random_state=42)
model_sklearn.fit(X_train, y_train)
sklearn_time = time.time() - start
preds_sklearn = model_sklearn.predict(X_test)
acc_sklearn = accuracy_score(y_test, preds_sklearn)
results.append(["Scikit-Learn", acc_sklearn, sklearn_time])


# ==========================================
# 4. PYTORCH IMPLEMENTATION
# ==========================================
class SVM_PyTorch(nn.Module):
    def __init__(self, n_features):
        super(SVM_PyTorch, self).__init__()
        # Single linear layer: y = wx + b
        self.linear = nn.Linear(n_features, 1) 
        
    def forward(self, x):
        return self.linear(x)

# Convert to Tensors
X_train_th = torch.tensor(X_train, dtype=torch.float32)
y_train_th = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_th = torch.tensor(X_test, dtype=torch.float32)

model_pt = SVM_PyTorch(X_train.shape[1])
optimizer = torch.optim.SGD(model_pt.parameters(), lr=0.01)

start = time.time()
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model_pt(X_train_th)
    
    # Hinge Loss Implementation in PyTorch
    # loss = mean(max(0, 1 - y * output))
    loss = torch.mean(torch.clamp(1 - y_train_th * outputs, min=0))
    # Add L2 Regularization manually (optional, or use weight_decay in optim)
    l2_lambda = 0.01
    l2_norm = sum(p.pow(2.0).sum() for p in model_pt.parameters())
    loss = loss + l2_lambda * l2_norm
    
    loss.backward()
    optimizer.step()

pt_time = time.time() - start

# Predict
with torch.no_grad():
    outputs_test = model_pt(X_test_th)
    preds_pt = torch.sign(outputs_test).flatten().numpy()
acc_pt = accuracy_score(y_test, preds_pt)
results.append(["PyTorch", acc_pt, pt_time])


# ==========================================
# 5. TENSORFLOW / KERAS IMPLEMENTATION
# ==========================================
# Using Keras Hinge Loss
model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# Use Hinge loss and SGD optimizer
model_tf.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), 
                 loss='hinge', 
                 metrics=['accuracy'])

start = time.time()
model_tf.fit(X_train, y_train, epochs=50, verbose=0, batch_size=32)
tf_time = time.time() - start

# Predict
outputs_tf = model_tf.predict(X_test, verbose=0)
preds_tf = np.sign(outputs_tf).flatten() # Sign function for binary class
acc_tf = accuracy_score(y_test, preds_tf)
results.append(["TensorFlow", acc_tf, tf_time])


# ==========================================
# 6. COMPARISON AND PREDICTION DEMO
# ==========================================
print("\n--- PERFORMANCE COMPARISON ---")
df_results = pd.DataFrame(results, columns=["Framework", "Accuracy", "Time (s)"])
print(df_results)

print("\n--- PREDICTION FUNCTION DEMO (First 5 Test Samples) ---")
print(f"Actual Labels: {y_test[:5]}")
print(f"Scratch Preds: {preds_scratch[:5].astype(int)}")
print(f"Sklearn Preds: {preds_sklearn[:5].astype(int)}")