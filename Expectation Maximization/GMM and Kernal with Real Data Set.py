import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 1. Load Real World Data (Iris)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using 2 features for 2D visualization
y_true = iris.target

# 2. Setup Metrics Tracking
k_range = range(1, 11)
bic_scores, aic_scores, inertia_scores = [], [], []

for k in k_range:
    # GMM Metrics
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))
    
    # K-Means Metric (Inertia/WCSS)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertia_scores.append(km.inertia_)

# 3. Visualization: Model Selection Criteria
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Plot A: Elbow Method
ax1.plot(k_range, inertia_scores, marker='o', color='blue')
ax1.set_title('Elbow Method (K-Means)')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (WCSS)')
ax1.grid(True, alpha=0.3)

# Plot B: BIC/AIC
ax2.plot(k_range, bic_scores, marker='o', label='BIC (Strict)')
ax2.plot(k_range, aic_scores, marker='s', label='AIC (Generous)')
ax2.set_title('BIC & AIC Scores (GMM)')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Score (Lower is better)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.show()

# 4. Side-by-Side Final Clustering (Using K=3 for comparison)
k_final = 3
km_final = KMeans(n_clusters=k_final, random_state=42, n_init=10).fit(X)
gmm_final = GaussianMixture(n_components=k_final, random_state=42).fit(X)

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# K-Means Result
ax3.scatter(X[:, 0], X[:, 1], c=km_final.labels_, cmap='viridis', edgecolors='k')
ax3.set_title(f"K-Means (ARI: {adjusted_rand_score(y_true, km_final.labels_):.3f})")

# GMM Result (Sizing by confidence)
probs = gmm_final.predict_proba(X)
labels = gmm_final.predict(X)
ax4.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s=probs.max(axis=1)*100)
ax4.set_title(f"GMM EM (ARI: {adjusted_rand_score(y_true, labels):.3f})")

plt.show()