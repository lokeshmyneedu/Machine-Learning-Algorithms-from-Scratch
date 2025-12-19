
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 1. Generate Data
# We create blobs and then stretch them to make them elliptical
X, y_true = make_blobs(n_samples=400, centers=2, cluster_std=0.60, random_state=0)
rng = np.random.RandomState(13)
transformation = [[0.6, -0.6], [-0.4, 0.8]] # Matrix to stretch the data
X_stretched = np.dot(X, transformation)

# 2. Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X_stretched)

# 3. Apply GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_stretched)

# 4. Visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot K-Means Results
ax[0].scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels_kmeans, cmap='viridis', s=40)
ax[0].set_title('K-Means Result\n(Fails: Cuts through the natural shape)')

# Plot GMM Results
ax[1].scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels_gmm, cmap='viridis', s=40)
ax[1].set_title('GMM Result\n(Success: Adapts to the oval shape)')

plt.show()


# Assuming X_stretched from the previous block

# 1. Check K-Means Quality
kmeans_score = silhouette_score(X_stretched, labels_kmeans)
print(f"K-Means Silhouette Score: {kmeans_score:.3f} (Higher is better)")

# 2. Check GMM Quality using BIC
# Note: We can't compare Silhouette directly to BIC, 
# but we can compare GMM BIC to a 'bad' model's BIC.
print(f"GMM BIC Score: {gmm.bic(X_stretched):.3f} (Lower is better)")