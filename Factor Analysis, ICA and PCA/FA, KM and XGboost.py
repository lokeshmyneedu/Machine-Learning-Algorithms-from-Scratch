import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Data
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# 2. Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Step A: Generate Factor Scores (Latent Variables)
fa = FactorAnalysis(n_components=3, random_state=42)
factor_scores = fa.fit_transform(X_scaled)
df_factors = pd.DataFrame(factor_scores, columns=['Factor_1', 'Factor_2', 'Factor_3'])

# 4. Step B: Generate Cluster Labels (Latent Classes)
# We use k=3 based on our previous Elbow/BIC analysis
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(factor_scores)

# 5. Step C: Combine everything into one "Super-Feature" Set
X_final = df.copy()
X_final['Factor_1'] = df_factors['Factor_1']
X_final['Factor_2'] = df_factors['Factor_2']
X_final['Cluster_Label'] = cluster_labels

# 6. Train/Test Split and Model
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Feature Importance: See if the Model actually liked our new features!
importances = pd.Series(clf.feature_importances_, index=X_final.columns)
print("\nTop 5 Most Important Features:")
print(importances.sort_values(ascending=False).head(5))