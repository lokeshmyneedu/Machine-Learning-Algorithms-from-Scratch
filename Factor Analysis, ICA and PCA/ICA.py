import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

# 1. Generate Synthetic 'Independent' Features
time = np.linspace(0, 8, 2000)
s1 = np.sin(2 * time)  # Signal 1: Sine wave
s2 = np.sign(np.sin(3 * time))  # Signal 2: Square wave
S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape)  # Add some noise

# 2. Mix them (Simulating 3 sensors picking up these 2 sources)
A = np.array([[1, 1], [0.5, 2], [1.5, 1.0]])  # Mixing matrix
X = S.dot(A.T)  # Mixed observations

# 3. Apply ICA
ica = FastICA(n_components=2, random_state=42)
S_ica = ica.fit_transform(X)  # Estimate the sources

# 4. Compare with PCA (to show why ICA is different)
pca = PCA(n_components=2)
S_pca = pca.fit_transform(X)

# 5. Visualization
models = [X, S, S_ica, S_pca]
names = ['Mixed Observations (Sensors)', 'True Sources', 
         'ICA Recovered Features', 'PCA Recovered Features']
colors = ['red', 'steelblue', 'orange']

plt.figure(figsize=(12, 8))
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.tight_layout()
plt.show()