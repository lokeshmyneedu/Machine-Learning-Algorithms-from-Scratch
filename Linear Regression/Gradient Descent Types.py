import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import time

class GradientDescentComparison:
    """Compare Batch, Stochastic, and Mini-Batch Gradient Descent."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialize with training data."""
        self.X = np.c_[np.ones(X.shape[0]), X]  # Add bias
        self.y = y
        self.m, self.n = self.X.shape
        
    def compute_cost(self, beta: np.ndarray) -> float:
        """Compute MSE cost."""
        predictions = self.X @ beta
        return (1 / (2 * self.m)) * np.sum((predictions - self.y) ** 2)
    
    def compute_gradient(self, beta: np.ndarray, indices: np.ndarray = None) -> np.ndarray:
        """
        Compute gradient.
        If indices is None, use all samples (batch).
        Otherwise, use specified samples.
        """
        if indices is None:
            X_batch = self.X
            y_batch = self.y
        else:
            X_batch = self.X[indices]
            y_batch = self.y[indices]
        
        m_batch = len(y_batch)
        predictions = X_batch @ beta
        gradient = (1 / m_batch) * X_batch.T @ (predictions - y_batch)
        return gradient
    
    def batch_gd(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Batch Gradient Descent.
        Uses ALL samples for each update.
        """
        beta = np.random.randn(self.n) * 0.01
        cost_history = []
        beta_history = []
        
        for i in range(n_iterations):
            # Compute gradient using ALL data
            gradient = self.compute_gradient(beta, indices=None)
            
            # Single update per iteration
            beta = beta - learning_rate * gradient
            
            # Track progress
            cost = self.compute_cost(beta)
            cost_history.append(cost)
            beta_history.append(beta.copy())
        
        return beta, cost_history, beta_history
    
    def stochastic_gd(self, learning_rate: float = 0.01, n_epochs: int = 100) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Stochastic Gradient Descent.
        Uses ONE random sample for each update.
        """
        beta = np.random.randn(self.n) * 0.01
        cost_history = []
        beta_history = []
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(self.m)
            
            # Update for each sample
            for idx in indices:
                # Compute gradient using ONE sample
                gradient = self.compute_gradient(beta, indices=[idx])
                
                # Immediate update
                beta = beta - learning_rate * gradient
                
                # Track progress (every sample)
                if len(cost_history) % 10 == 0:  # Don't track every update (too many)
                    cost = self.compute_cost(beta)
                    cost_history.append(cost)
                    beta_history.append(beta.copy())
        
        return beta, cost_history, beta_history
    
    def mini_batch_gd(self, learning_rate: float = 0.01, n_epochs: int = 100, 
                      batch_size: int = 32) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Mini-Batch Gradient Descent.
        Uses small batches for each update.
        """
        beta = np.random.randn(self.n) * 0.01
        cost_history = []
        beta_history = []
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(self.m)
            
            # Process mini-batches
            for i in range(0, self.m, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Compute gradient on mini-batch
                gradient = self.compute_gradient(beta, indices=batch_indices)
                
                # Update
                beta = beta - learning_rate * gradient
                
                # Track progress
                cost = self.compute_cost(beta)
                cost_history.append(cost)
                beta_history.append(beta.copy())
        
        return beta, cost_history, beta_history


def example1_convergence_comparison():
    """Compare convergence paths of all three methods."""
    print("=" * 80)
    print("EXAMPLE 1: Convergence Comparison")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(200, 1)
    y = 4 + 3 * X.ravel() + np.random.randn(200) * 0.5
    
    # Initialize comparison
    gd_comp = GradientDescentComparison(X, y)
    
    # Run all three methods
    print("\n1. Running Batch Gradient Descent...")
    start = time.time()
    beta_batch, cost_batch, _ = gd_comp.batch_gd(learning_rate=0.01, n_iterations=1000)
    time_batch = time.time() - start
    print(f"   Time: {time_batch:.3f}s, Final cost: {cost_batch[-1]:.6f}")
    
    print("\n2. Running Stochastic Gradient Descent...")
    start = time.time()
    beta_sgd, cost_sgd, _ = gd_comp.stochastic_gd(learning_rate=0.001, n_epochs=10)
    time_sgd = time.time() - start
    print(f"   Time: {time_sgd:.3f}s, Final cost: {cost_sgd[-1]:.6f}")
    
    print("\n3. Running Mini-Batch Gradient Descent...")
    start = time.time()
    beta_mb, cost_mb, _ = gd_comp.mini_batch_gd(learning_rate=0.01, n_epochs=100, batch_size=32)
    time_mb = time.time() - start
    print(f"   Time: {time_mb:.3f}s, Final cost: {cost_mb[-1]:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cost vs Iterations (log scale)
    axes[0, 0].plot(cost