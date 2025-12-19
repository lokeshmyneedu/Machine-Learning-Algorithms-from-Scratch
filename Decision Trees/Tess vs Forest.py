import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Generate Non-Linear Data (The "Moons")
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Train a Single Decision Tree (Prone to Overfitting)
tree_clf = DecisionTreeClassifier(random_state=42) # No max_depth limit!
tree_clf.fit(X_train, y_train)

# 3. Train a Random Forest (The Ensemble)
# n_estimators=500 -> We are building a committee of 500 trees
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)

# 4. Check Accuracy
print(f"Single Tree Accuracy:   {tree_clf.score(X_test, y_test):.4f}")
print(f"Random Forest Accuracy: {rf_clf.score(X_test, y_test):.4f}")

# 5. Visualization Comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot Single Tree
DecisionBoundaryDisplay.from_estimator(tree_clf, X, cmap=plt.cm.RdYlBu, ax=ax[0], alpha=0.8)
ax[0].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
ax[0].set_title("Single Decision Tree\n(High Variance, Jagged)")

# Plot Random Forest
DecisionBoundaryDisplay.from_estimator(rf_clf, X, cmap=plt.cm.RdYlBu, ax=ax[1], alpha=0.8)
ax[1].scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
ax[1].set_title("Random Forest (500 Trees)\n(Low Variance, Smooth)")

plt.show()