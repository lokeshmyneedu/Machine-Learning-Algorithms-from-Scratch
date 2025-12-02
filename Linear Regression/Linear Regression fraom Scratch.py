import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class LinearRegression:
    """
    Linear Regression implementation from scratch with multiple optimization methods.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 regularization: Optional[str] = None, lambda_reg: float = 0.01):
        """
        Initialize Linear Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        regularization : str, optional
            Type of regularization ('l1', 'l2', or None)
        lambda_reg : float
            Regularization parameter
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to feature matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost function.
        J(β) = (1/2m) * Σ(h(x) - y)² + regularization_term
        """
        m = len(y)
        predictions = X @ weights
        mse = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        
        # Add regularization term (exclude bias term from regularization)
        if self.regularization == 'l2':  # Ridge
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)
            return mse + reg_term
        elif self.regularization == 'l1':  # Lasso
            reg_term = (self.lambda_reg / m) * np.sum(np.abs(weights[1:]))
            return mse + reg_term
        
        return mse
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cost function.
        ∇J(β) = (1/m) * X^T(Xβ - y) + regularization_gradient
        """
        m = len(y)
        predictions = X @ weights
        gradient = (1 / m) * X.T @ (predictions - y)
        
        # Add regularization gradient (exclude bias term)
        if self.regularization == 'l2':  # Ridge
            reg_gradient = np.copy(gradient)
            reg_gradient[1:] += (self.lambda_reg / m) * weights[1:]
            return reg_gradient
        elif self.regularization == 'l1':  # Lasso
            reg_gradient = np.copy(gradient)
            reg_gradient[1:] += (self.lambda_reg / m) * np.sign(weights[1:])
            return reg_gradient
        
        return gradient
    
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Fit model using batch gradient descent.
        
        Update rule: β := β - α * ∇J(β)
        """
        X_with_bias = self._add_bias(X)
        m, n = X_with_bias.shape
        
        # Initialize weights randomly
        self.weights = np.random.randn(n) * 0.01
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Compute gradient
            gradient = self._compute_gradient(X_with_bias, y, self.weights)
            
            # Update weights
            self.weights -= self.lr * gradient
            
            # Compute and store cost
            cost = self._compute_cost(X_with_bias, y, self.weights)
            self.cost_history.append(cost)
            
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        # Separate bias and weights
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
        return self
    
    def fit_sgd(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                verbose: bool = True):
        """
        Fit model using Stochastic Gradient Descent (mini-batch).
        
        Updates weights using small batches of data for faster convergence.
        """
        X_with_bias = self._add_bias(X)
        m, n = X_with_bias.shape
        
        # Initialize weights
        self.weights = np.random.randn(n) * 0.01
        self.cost_history = []
        
        for epoch in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            # Mini-batch updates
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                gradient = self._compute_gradient(X_batch, y_batch, self.weights)
                self.weights -= self.lr * gradient
            
            # Compute cost on full dataset
            cost = self._compute_cost(X_with_bias, y, self.weights)
            self.cost_history.append(cost)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
        return self
    
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model using normal equation (closed-form solution).
        
        β = (X^T X)^(-1) X^T y  (without regularization)
        β = (X^T X + λI)^(-1) X^T y  (with L2 regularization)
        """
        X_with_bias = self._add_bias(X)
        
        if self.regularization == 'l2':  # Ridge regression
            # Add regularization term to X^T X (exclude bias from regularization)
            n = X_with_bias.shape[1]
            identity = np.eye(n)
            identity[0, 0] = 0  # Don't regularize bias term
            
            self.weights = np.linalg.inv(
                X_with_bias.T @ X_with_bias + self.lambda_reg * identity
            ) @ X_with_bias.T @ y
        else:
            # Standard normal equation
            self.weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).
        R² = 1 - (SS_res / SS_tot)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class FeatureScaler:
    """Feature scaling using standardization (z-score normalization)."""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray):
        """Compute mean and standard deviation."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to have mean=0 and std=1."""
        return (X - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert scaled features back to original scale."""
        return X * self.std + self.mean


# ==================== EXAMPLE USAGE ====================

def generate_sample_data(n_samples: int = 100, noise: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for demonstration."""
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X.squeeze() + noise * np.random.randn(n_samples)
    return X, y


def example_simple_regression():
    """Example 1: Simple Linear Regression with Gradient Descent."""
    print("=" * 60)
    print("EXAMPLE 1: Simple Linear Regression (Gradient Descent)")
    print("=" * 60)
    
    # Generate data
    X, y = generate_sample_data(n_samples=100, noise=5)
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit_gradient_descent(X, y, verbose=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluate
    r2 = model.score(X, y)
    print(f"\nFinal Model: y = {model.bias:.2f} + {model.weights[0]:.2f}x")
    print(f"R² Score: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Data and predictions
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X, y_pred, 'r-', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    
    # Plot 2: Cost history
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function over Iterations')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_example1.png', dpi=100, bbox_inches='tight')
    plt.show()


def example_normal_equation():
    """Example 2: Normal Equation vs Gradient Descent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Normal Equation vs Gradient Descent")
    print("=" * 60)
    
    X, y = generate_sample_data(n_samples=100, noise=5)
    
    # Method 1: Gradient Descent
    model_gd = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model_gd.fit_gradient_descent(X, y, verbose=False)
    
    # Method 2: Normal Equation
    model_ne = LinearRegression()
    model_ne.fit_normal_equation(X, y)
    
    print(f"\nGradient Descent: y = {model_gd.bias:.4f} + {model_gd.weights[0]:.4f}x")
    print(f"Normal Equation:  y = {model_ne.bias:.4f} + {model_ne.weights[0]:.4f}x")
    print(f"\nGD R² Score: {model_gd.score(X, y):.4f}")
    print(f"NE R² Score: {model_ne.score(X, y):.4f}")


def example_regularization():
    """Example 3: Ridge Regression (L2 Regularization)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Ridge Regression (L2 Regularization)")
    print("=" * 60)
    
    # Generate data with multiple features
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = 2 + 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + np.random.randn(50) * 0.5
    
    # Train models with different regularization
    lambdas = [0, 0.1, 1.0, 10.0]
    
    plt.figure(figsize=(12, 4))
    
    for idx, lam in enumerate(lambdas):
        model = LinearRegression(learning_rate=0.01, n_iterations=1000,
                                regularization='l2', lambda_reg=lam)
        model.fit_gradient_descent(X, y, verbose=False)
        
        print(f"\nλ = {lam}:")
        print(f"  Weights: {model.weights}")
        print(f"  R² Score: {model.score(X, y):.4f}")
        
        # Plot cost history
        plt.subplot(1, 4, idx + 1)
        plt.plot(model.cost_history)
        plt.title(f'λ = {lam}')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_regularization.png', dpi=100, bbox_inches='tight')
    plt.show()


def example_multiple_features():
    """Example 4: Multiple Linear Regression with Feature Scaling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multiple Linear Regression with Feature Scaling")
    print("=" * 60)
    
    # Generate multi-feature data
    np.random.seed(42)
    X = np.random.rand(100, 3) * 100  # Features with different scales
    y = 50 + 2*X[:, 0] + 0.5*X[:, 1] - 1.5*X[:, 2] + np.random.randn(100) * 5
    
    # Without scaling
    model_unscaled = LinearRegression(learning_rate=0.0001, n_iterations=1000)
    model_unscaled.fit_gradient_descent(X, y, verbose=False)
    
    # With scaling
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)
    model_scaled = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model_scaled.fit_gradient_descent(X_scaled, y, verbose=False)
    
    print(f"\nUnscaled R²: {model_unscaled.score(X, y):.4f}")
    print(f"Scaled R²:   {model_scaled.score(X_scaled, y):.4f}")
    
    # Plot cost histories
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model_unscaled.cost_history)
    plt.title('Without Feature Scaling')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model_scaled.cost_history)
    plt.title('With Feature Scaling')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_scaling.png', dpi=100, bbox_inches='tight')
    plt.show()


# Run all examples
if __name__ == "__main__":
    example_simple_regression()
    example_normal_equation()
    example_regularization()
    example_multiple_features()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)