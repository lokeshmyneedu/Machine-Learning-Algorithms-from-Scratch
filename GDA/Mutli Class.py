from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import numpy as np

# 1. Create dummy data with 3 classes (0, 1, 2)
# Features: [Height, Weight]
X = np.array([
    [1, 2], [2, 1],       # Class 0
    [5, 5], [6, 4],       # Class 1
    [9, 10], [10, 8]      # Class 2
])
y = np.array([0, 0, 1, 1, 2, 2]) # 3 classes

# 2. Fit GDA (LDA)
lda = LinearDiscriminantAnalysis() # Native multi-class
lda.fit(X, y)

# 3. Fit Naive Bayes
nb = GaussianNB() # Native multi-class
nb.fit(X, y)

# 4. Predict a new point (e.g., somewhere between class 1 and 2)
new_point = [[8, 8]]

print(f"LDA Prediction: {lda.predict(new_point)}")
print(f"NB Prediction:  {nb.predict(new_point)}")

# You can even see the probability scores for all 3 classes
print(f"LDA Probabilities (Class 0, 1, 2): \n{lda.predict_proba(new_point)}")