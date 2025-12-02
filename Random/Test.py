import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Define the Non-Convex Function and its Derivative
def f(x):
    return x**4 - 4*x**2 + x

def df(x):
    return 4*x**3 - 8*x + 1

# 2. Configuration
x_start = -2.4  # Starting point (far left)
learning_rate = 0.01
iterations = 100

# 3. Implement the Optimizers

# A. Standard Gradient Descent
path_sgd = [x_start]
x = x_start
for _ in range(iterations):
    grad = df(x)
    x = x - learning_rate * grad
    path_sgd.append(x)

# B. Momentum
path_momentum = [x_start]
x = x_start
velocity = 0
gamma = 0.9  # Momentum factor (friction)
for _ in range(iterations):
    grad = df(x)
    # Update velocity: keep 90% of old speed, add current gradient
    velocity = gamma * velocity + learning_rate * grad
    x = x - velocity
    path_momentum.append(x)

# C. Adam (Simplified implementation)
path_adam = [x_start]
x = x_start
m = 0  # First moment (momentum)
v = 0  # Second moment (variance)
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
for t in range(1, iterations + 1):
    grad = df(x)
    
    # Update moments
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    # Bias correction (warm-up)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # Update parameter
    x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    path_adam.append(x)

# 4. Visualization
x_grid = np.linspace(-3, 3, 400)
y_grid = f(x_grid)

plt.figure(figsize=(10, 6))
plt.plot(x_grid, y_grid, 'k-', alpha=0.6, label='Loss Landscape')
plt.title('Comparison: SGD vs Momentum vs Adam on Non-Convex Function')
plt.xlabel('Parameter Value (x)')
plt.ylabel('Loss (Error)')

# Plot final paths
plt.plot(path_sgd, [f(p) for p in path_sgd], 'ro-', label='SGD (Stuck)', alpha=0.5, markersize=5)
plt.plot(path_momentum, [f(p) for p in path_momentum], 'bo-', label='Momentum (Overshoots)', alpha=0.5, markersize=5)
plt.plot(path_adam, [f(p) for p in path_adam], 'go-', label='Adam (Efficient)', alpha=0.5, markersize=5)

# Highlight the trap and the goal
plt.scatter([-1.44], [f(-1.44)], color='red', s=100, zorder=5, marker='x', label='Local Min (Trap)')
plt.scatter([1.38], [f(1.38)], color='green', s=100, zorder=5, marker='*', label='Global Min (Goal)')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()