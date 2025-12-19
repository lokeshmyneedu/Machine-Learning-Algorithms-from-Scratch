from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Load data
data = load_iris()
X, y = data.data, data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the model (using K=3)
model = KNeighborsClassifier(n_neighbors=3)

# 4. "Train" the model (KNN just stores the data here)
model.fit(X_train, y_train)

# 5. Make a prediction
prediction = model.predict(X_test[:1])
print(f"Predicted class: {prediction}")