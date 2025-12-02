import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# 1. Generate Synthetic Data with Correlation
# We manually create data to ensure specific covariance structures (rotated blobs)
# This highlights where QDA succeeds and Naive Bayes fails
np.random.seed(42)

# Class 0: Rotated blob
mean0 = [2, 2]
cov0 = [[1.5, 1.0], [1.0, 1.5]]  # Diagonal covariance (correlated features)
x0 = np.random.multivariate_normal(mean0, cov0, 200)
y0 = np.zeros(200)

# Class 1: Different rotated blob
mean1 = [-1, -1]
cov1 = [[2.0, -0.8], [-0.8, 2.0]] # Different covariance
x1 = np.random.multivariate_normal(mean1, cov1, 200)
y1 = np.ones(200)

# Combine
X = np.concatenate((x0, x1))
y = np.concatenate((y0, y1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize Models
models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "LDA (Linear Discriminant)": LinearDiscriminantAnalysis(),
    "QDA (Quadratic Discriminant)": QuadraticDiscriminantAnalysis()
}

# 3. Setup Plotting Mesh
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 4. Plot
plt.figure(figsize=(18, 5))

for i, (name, clf) in enumerate(models.items()):
    # Fit the model (This performs the MLE calculations derived previously)
    clf.fit(X_train, y_train)
    
    # Predict on the meshgrid to visualize the decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax = plt.subplot(1, 3, i + 1)
    
    # Plot contour (Decision Boundary)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    # Plot data points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
    
    # Calculate accuracy for display
    score = clf.score(X_test, y_test)
    
    ax.set_title(f"{name}\nAccuracy: {score:.2f}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()