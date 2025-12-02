import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# 1. Generate synthetic 2D data
# We use 2 features so we can plot them on X and Y axes
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, 
                           flip_y=0.05, class_sep=1.5, random_state=42)

# 2. Initialize and Fit Models
models = [
    ("LDA (Linear Boundary)", LinearDiscriminantAnalysis()),
    ("QDA (Quadratic Boundary)", QuadraticDiscriminantAnalysis())
]

# Fit the models
for name, model in models:
    model.fit(X, y)

# 3. Set up the plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create a meshgrid to plot decision boundaries
# This creates a grid of points covering the range of our data
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

for i, (name, model) in enumerate(models):
    # Predict the class for every point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour (the background colors)
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Scatter plot the actual data points
    # Class 0 points
    axes[i].scatter(X[y == 0, 0], X[y == 0, 1], c='blue', 
                    edgecolor='k', s=20, label='Class 0')
    # Class 1 points
    axes[i].scatter(X[y == 1, 0], X[y == 1, 1], c='red', 
                    edgecolor='k', s=20, label='Class 1')
    
    axes[i].set_title(name, fontsize=14)
    axes[i].legend()

plt.tight_layout()
plt.show()