import numpy as np
import matplotlib.pyplot as plt

# Parameters: Mean = 0, Standard Deviation = 1
mu, sigma = 0, 1 
samples = np.random.normal(mu, sigma, 10000)

# Plotting the histogram to see the 'Bell'
plt.hist(samples, bins=100, density=True, alpha=0.6, color='g')

# Plotting the theoretical curve
x = np.linspace(-4, 4, 100)
p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Gaussian (Normal) Distribution")
plt.show()