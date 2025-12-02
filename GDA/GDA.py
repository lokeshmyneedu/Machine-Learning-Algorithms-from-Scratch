import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. Feature Selection from Scratch
# ==========================================
class FisherSelector:
    """
    Selects features based on Fisher's Discriminant Ratio (Fisher Score).
    Measures the separation between class means normalized by variance.
    """
    def __init__(self, k=5):
        self.k = k
        self.selected_indices = None

    def fit(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)
        scores = []

        # Calculate Fisher Score for each feature
        for i in range(n_features):
            feature = X[:, i]
            # Mean and variance for Class 0
            m0 = np.mean(feature[y == classes[0]])
            v0 = np.var(feature[y == classes[0]])
            # Mean and variance for Class 1
            m1 = np.mean(feature[y == classes[1]])
            v1 = np.var(feature[y == classes[1]])
            
            # Fisher Score formula: (mu1 - mu2)^2 / (var1 + var2)
            score = ((m0 - m1)**2) / (v0 + v1 + 1e-6) # add epsilon for stability
            scores.append(score)

        # Select top k indices
        self.selected_indices = np.argsort(scores)[::-1][:self.k]
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

# ==========================================
# 2. Regularized GDA from Scratch
# ==========================================
class RegularizedGDA:
    """
    Gaussian Discriminant Analysis with Regularization (Covariance Shrinkage).
    """
    def __init__(self, reg_param=0.1):
        """
        reg_param (float): Shrinkage parameter (alpha).
                           0 = No regularization (Standard GDA/LDA)
                           1 = Full assumption of independence (Naive Bayes behavior)
        """
        self.reg_param = reg_param
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. Calculate Bernoulli prior (Phi)
        self.phi = np.mean(y) # P(Y=1)
        
        # 2. Separate data by class
        X0 = X[y == 0]
        X1 = X[y == 1]
        
        # 3. Calculate Means (Mu)
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)
        
        # 4. Calculate Covariance Matrix (Sigma)
        # In standard GDA (Linear Discriminant), classes share a Covariance Matrix
        # Calculate scatter matrices
        X0_centered = X0 - self.mu0
        X1_centered = X1 - self.mu1
        
        # Shared Covariance (unbiased estimator)
        sigma_sum = (X0_centered.T @ X0_centered) + (X1_centered.T @ X1_centered)
        self.sigma = sigma_sum / n_samples

        # 5. Regularization (Shrinkage)
        # We shrink the covariance matrix towards the Identity matrix
        # Sigma_reg = (1 - alpha) * Sigma + alpha * I
        identity = np.eye(n_features)
        self.sigma = (1 - self.reg_param) * self.sigma + self.reg_param * identity

    def predict_proba(self, X):
        # We compute the log-likelihood ratio to avoid numerical underflow
        # P(Y=1|X) / P(Y=0|X)
        
        n_samples = X.shape[0]
        inv_sigma = np.linalg.pinv(self.sigma) # Pseudo-inverse for stability
        
        preds = []
        
        # Constant terms (cancel out in comparison usually, but needed for probability)
        # However, a simpler way for binary classification is the Linear Decision Boundary formula:
        # theta^T * x + theta_0 > 0
        
        # Let's use the exact PDF definitions for clarity and robustness
        det_sigma = np.linalg.det(self.sigma)
        if det_sigma <= 0: det_sigma = 1e-6 # Avoid log(0)
        
        const = 1 / ((2 * np.pi) ** (X.shape[1] / 2) * (det_sigma ** 0.5))
        
        probs = []
        for x in X:
            # P(X|Y=0)
            d0 = x - self.mu0
            p_x_y0 = const * np.exp(-0.5 * d0.T @ inv_sigma @ d0)
            
            # P(X|Y=1)
            d1 = x - self.mu1
            p_x_y1 = const * np.exp(-0.5 * d1.T @ inv_sigma @ d1)
            
            # Bayes Rule: P(Y=1|X) = (P(X|Y=1) * P(Y=1)) / P(X)
            # P(X) = P(X|Y=1)P(Y=1) + P(X|Y=0)P(Y=0)
            numerator = p_x_y1 * self.phi
            denominator = (p_x_y1 * self.phi) + (p_x_y0 * (1 - self.phi))
            
            probs.append(numerator / (denominator + 1e-9))
            
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ==========================================
# 3. Real World Experiment: Breast Cancer Data
# ==========================================

# Load Data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"Dataset Shape: {X.shape}")
print(f"Classes: {data.target_names}")

# Split Data (Using a smaller train set to highlight GDA data efficiency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale Data (Important for Regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- A. Apply Custom Feature Selection ---
print("\n--- Step 1: Feature Selection (Fisher Score) ---")
selector = FisherSelector(k=10) # Select top 10 features
selector.fit(X_train_scaled, y_train)

print(f"Selected Features: {feature_names[selector.selected_indices]}")

X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

# --- B. Train Custom Regularized GDA ---
print("\n--- Step 2: Training Regularized GDA (Generative) ---")
gda = RegularizedGDA(reg_param=0.1) # 10% shrinkage
gda.fit(X_train_sel, y_train)
y_pred_gda = gda.predict(X_test_sel)
acc_gda = accuracy_score(y_test, y_pred_gda)

# --- C. Train Logistic Regression (Discriminative) ---
print("--- Step 3: Training Logistic Regression (Discriminative) ---")
lr = LogisticRegression(random_state=42)
lr.fit(X_train_sel, y_train)
y_pred_lr = lr.predict(X_test_sel)
acc_lr = accuracy_score(y_test, y_pred_lr)

# ==========================================
# 4. Comparison Results
# ==========================================
print("\n" + "="*40)
print("FINAL RESULTS")
print("="*40)
print(f"Custom GDA Accuracy:        {acc_gda:.4f}")
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")

if acc_gda > acc_lr:
    print("\nCONCLUSION: GDA outperformed Logistic Regression.")
    print("Reason: The data likely follows a Gaussian distribution closely, allowing GDA")
    print("to capture the structure more efficiently with the available data.")
elif acc_gda == acc_lr:
     print("\nCONCLUSION: Both models performed equally.")
else:
    print("\nCONCLUSION: Logistic Regression outperformed GDA.")
    print("Reason: The Gaussian assumption may be violated (outliers/non-normal data),")
    print("allowing the more robust discriminative model (LR) to win.")