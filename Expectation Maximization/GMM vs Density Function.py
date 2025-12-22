import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# 1. Simple Density Function (Single Gaussian)
def simple_gaussian_detection(data, point, threshold=0.01):
    mu = np.mean(data, axis=0)
    sigma = np.cov(data, rowvar=False)
    # Calculate probability density
    p = multivariate_normal.pdf(point, mean=mu, cov=sigma)
    return "Anomaly" if p < threshold else "Normal"

# 2. Gaussian Mixture Model (GMM)
def gmm_detection(data, point, threshold=-5.0):
    # Fit GMM with 3 components (clusters)
    gmm = GaussianMixture(n_components=3, covariance_type='full')
    gmm.fit(data)
    
    # GMM uses log-likelihood for stability
    score = gmm.score_samples(point.reshape(1, -1))
    return "Anomaly" if score < threshold else "Normal"