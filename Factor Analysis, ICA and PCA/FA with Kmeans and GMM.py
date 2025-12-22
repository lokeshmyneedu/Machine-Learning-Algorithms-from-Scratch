import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# 1. Load and Prep
data = load_wine()
X_scaled = StandardScaler().fit_transform(data.data)

# 2. Factor Analysis (Feature Engineering)
# We condense 13 features into 3 Latent Factors
fa = FactorAnalysis(n_components=3, random_state=42)
X_factors = fa.fit_transform(X_scaled)

# 3. Iterate to find optimal Clusters (k)
n_components = np.arange(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=42).fit(X_factors) 
          for n in n_components]

# 4. Extract AIC and BIC
aic_values = [m.aic(X_factors) for m in models]
bic_values = [m.bic(X_factors) for m in models]

# 5. Plotting (The "Elbow" will show k=3 is best)
plt.plot(n_components, aic_values, label='AIC')
plt.plot(n_components, bic_values, label='BIC')
plt.legend()
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.show()

# 6. K-Means Elbow Method (on Factor Scores)
wcss = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_factors)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_factors, kmeans.labels_))

# 7. Visualization
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot WCSS (Elbow)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('WCSS (Inertia)', color='tab:blue')
ax1.plot(k_range, wcss, color='tab:blue', marker='o', label='WCSS')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot Silhouette Score on same graph
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='tab:red')
ax2.plot(k_range, sil_scores, color='tab:red', marker='s', label='Silhouette')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('K-Means Elbow & Silhouette Analysis (on Factor Scores)')
plt.grid(True)
plt.show()