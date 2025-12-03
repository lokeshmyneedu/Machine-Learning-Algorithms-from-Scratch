import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Generate Non-Linear Data (The "Moons" dataset)
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Scale data (CRITICAL for SVMs to work correctly)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Linear SVM": SVC(kernel='linear', C=1.0),
    "Complex SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma=1.0)
}

# 3. Visualization Setup
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
h = .02  # Step size in the mesh

for i, (name, model) in enumerate(models.items()):
    clf = model.fit(X_scaled, y)
    
    # Create a mesh to plot decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for the whole mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and training points
    axes[i].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    axes[i].set_title(name)
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')

plt.suptitle('Classification Boundaries: Linear vs. Complex', fontsize=16)
plt.show()