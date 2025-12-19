from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load Data
data = load_breast_cancer()
X = data.data    # Features (30 different measurements)
y = data.target  # Labels (0 = Malignant, 1 = Benign)

# 2. Split Data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Decision Tree
# We limit max_depth=3 to keep the tree understandable (prevents overfitting)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy on Test Data: {acc * 100:.2f}%")

# 5. Visualize the Logic
plt.figure(figsize=(15,8))
plot_tree(clf, 
          feature_names=data.feature_names, 
          class_names=data.target_names, 
          filled=True, 
          fontsize=10)
plt.title("Decision Tree Logic for Breast Cancer Detection")
plt.show()