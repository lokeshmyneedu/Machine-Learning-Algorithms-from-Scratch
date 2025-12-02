import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# 1. Create simple data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# 2. Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X, y)

# 3. Train Linear SVM
svm_model = LinearSVC(dual="auto", random_state=42)
svm_model.fit(X, y)

# --- THE BIG DIFFERENCE ---

# Logistic Regression gives probabilities naturally
print("Logistic Probabilities (First 3):")
print(lr_model.predict_proba(X[:3])) 
# Output: [[0.03, 0.97], [0.88, 0.12], ...] (Clear confidence)

# SVM strictly gives labels. This line would crash:
# print(svm_model.predict_proba(X[:3])) -> AttributeError: has no attribute 'predict_proba'

# To get probabilities from SVM, you must "hack" it using calibration
svm_calibrated = CalibratedClassifierCV(svm_model) 
svm_calibrated.fit(X, y)
print("\nSVM 'Probabilities' (Calibrated):")
print(svm_calibrated.predict_proba(X[:3]))