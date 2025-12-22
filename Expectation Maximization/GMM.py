import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 1. Generate synthetic data with 3 clusters
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=42)

# 2. Setup GMM
# warm_start=True allows us to call .fit() repeatedly to see step-by-step progress
gmm = GaussianMixture(n_components=3, max_iter=1, warm_start=True, random_state=42)

log_likelihoods = []
iterations = 20

# 3. Manually run iterations to capture the log-likelihood at each step
for i in range(iterations):
    gmm.fit(X)
    # .lower_bound_ is the ELBO value scikit-learn uses to check convergence
    log_likelihoods.append(gmm.lower_bound_)

# 4. Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(range(1, iterations + 1), log_likelihoods, marker='o', color='darkorange')
plt.title("EM Convergence: Log-Likelihood vs Iterations")
plt.xlabel("Iteration Number")
plt.ylabel("Log-Likelihood (Lower Bound)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

print(f"Final Convergence state: {gmm.converged_}")