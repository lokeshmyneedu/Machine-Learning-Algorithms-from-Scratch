import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 1. Load Real-World Data (Iris Dataset)
# We only take the first two features (Sepal length, Sepal width) for visualization
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

# 2. Create two models for comparison
# Model A: Linear Kernel (The "Straight Cut")
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X, y)

# Model B: RBF Kernel (The "Kernel Trick")
# gamma=0.7 controls the "curviness" or influence range of a single point
rbf_svm = SVC(kernel='rbf', gamma=0.7, C=1.0)
rbf_svm.fit(X, y)

# 3. Visualization Helper Function
def plot_decision_boundary(model, ax, title):
    # Create a grid of points to map the decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict on the entire grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and training points
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    
    # HIGHLIGHTING SUPPORT VECTORS (The "Remembered" Data)
    support_vectors = model.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    
    ax.set_title(title)
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')

# 4. Plotting
fig, sub = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_boundary(linear_svm, sub[0], "Linear Kernel (Straight Line)")
plot_decision_boundary(rbf_svm, sub[1], "RBF Kernel (Non-Linear/Curved)")

plt.legend(loc="upper right")
plt.show()

# 5. Proof of "Remembered Data"
print(f"Total training data points: {len(X)}")
print(f"Points 'remembered' by Linear Model (Support Vectors): {len(linear_svm.support_vectors_)}")
print(f"Points 'remembered' by RBF Model (Support Vectors): {len(rbf_svm.support_vectors_)}")