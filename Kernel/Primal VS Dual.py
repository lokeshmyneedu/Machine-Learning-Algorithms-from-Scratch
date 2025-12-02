"""
SVM: PRIMAL vs DUAL FORMULATION
================================
A comprehensive comparison explaining when and why to use each approach.

This script demonstrates:
1. Mathematical foundations of both formulations
2. Implementation of both approaches
3. When to use Primal vs Dual
4. The "Kernel Trick" and why it requires Dual
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


# ============================================================================
# PART 1: THEORETICAL EXPLANATION
# ============================================================================

def explain_primal_vs_dual():
    """
    Detailed explanation of Primal vs Dual formulations
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SVM: PRIMAL vs DUAL FORMULATION                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE ORIGINAL SVM PROBLEM (What we want to solve):                           ║
║  ─────────────────────────────────────────────────                           ║
║  Find a hyperplane w·x + b = 0 that:                                         ║
║  • Correctly classifies all points (or most, with soft margin)              ║
║  • Maximizes the margin (distance between classes)                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │                        PRIMAL FORMULATION                              │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  OPTIMIZATION PROBLEM:                                                       ║
║  ─────────────────────                                                       ║
║                     1                                                        ║
║     minimize      ─── ||w||² + C Σ ξᵢ                                       ║
║      w, b, ξ       2                                                         ║
║                                                                              ║
║     subject to:   yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ    for all i                       ║
║                   ξᵢ ≥ 0                                                     ║
║                                                                              ║
║  VARIABLES:                                                                  ║
║  ──────────                                                                  ║
║  • w ∈ ℝᵈ         (weight vector, d = number of features)                   ║
║  • b ∈ ℝ          (bias term)                                               ║
║  • ξᵢ ≥ 0         (slack variables for soft margin)                         ║
║                                                                              ║
║  COMPLEXITY: O(d) variables where d = number of features                     ║
║                                                                              ║
║  PREDICTION:  sign(w·x + b)                                                  ║
║                                                                              ║
║  PROS:                                                                       ║
║  • Efficient when n >> d (many samples, few features)                       ║
║  • Direct access to weight vector w                                         ║
║  • Can use gradient descent optimization                                    ║
║                                                                              ║
║  CONS:                                                                       ║
║  • CANNOT use kernels directly (no kernel trick)                            ║
║  • Inefficient when d >> n (many features, few samples)                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │                         DUAL FORMULATION                               │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  OPTIMIZATION PROBLEM:                                                       ║
║  ─────────────────────                                                       ║
║                                    1                                         ║
║     maximize      Σ αᵢ  -  ─── Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ · xⱼ)                ║
║        α                    2                                                ║
║                                                                              ║
║     subject to:   0 ≤ αᵢ ≤ C       for all i                                ║
║                   Σ αᵢ yᵢ = 0                                                ║
║                                                                              ║
║  VARIABLES:                                                                  ║
║  ──────────                                                                  ║
║  • αᵢ ∈ [0, C]    (Lagrange multipliers, one per sample)                    ║
║                                                                              ║
║  COMPLEXITY: O(n) variables where n = number of samples                      ║
║                                                                              ║
║  KEY INSIGHT: Data only appears as dot products (xᵢ · xⱼ)                   ║
║               This enables the KERNEL TRICK!                                 ║
║                                                                              ║
║  PREDICTION:  sign( Σᵢ αᵢ yᵢ K(xᵢ, x) + b )                                ║
║               (only support vectors with αᵢ > 0 contribute)                 ║
║                                                                              ║
║  PROS:                                                                       ║
║  • Enables KERNEL TRICK (non-linear boundaries)                             ║
║  • Efficient when d >> n (many features, few samples)                       ║
║  • Solution is sparse (only support vectors matter)                         ║
║                                                                              ║
║  CONS:                                                                       ║
║  • Slower when n >> d (many samples)                                        ║
║  • Requires storing kernel matrix O(n²)                                     ║
║  • More complex optimization (quadratic programming)                        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │                        THE KERNEL TRICK                                │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
║  WHY DUAL ENABLES KERNELS:                                                   ║
║  ─────────────────────────                                                   ║
║                                                                              ║
║  In Dual, data appears ONLY as dot products: (xᵢ · xⱼ)                      ║
║                                                                              ║
║  We can REPLACE this with a kernel function: K(xᵢ, xⱼ)                      ║
║                                                                              ║
║  The kernel computes dot products in a HIGHER-DIMENSIONAL space             ║
║  WITHOUT explicitly transforming the data!                                  ║
║                                                                              ║
║  EXAMPLE - RBF Kernel:                                                       ║
║  ─────────────────────                                                       ║
║  K(x, y) = exp(-γ ||x - y||²)                                               ║
║                                                                              ║
║  This is equivalent to mapping data to INFINITE dimensions,                 ║
║  but we never actually compute the transformation!                          ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  Original 2D Space              Implicit High-D Space               │    ║
║  │  ══════════════════             ════════════════════                │    ║
║  │                                                                     │    ║
║  │      ○  ○                            Linear                         │    ║
║  │    ○  ●●  ○     K(x,y)              Separation                      │    ║
║  │      ○  ○      ──────────►          Possible!                       │    ║
║  │                                                                     │    ║
║  │  (Not linearly                   (Data becomes                      │    ║
║  │   separable)                      linearly separable)               │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  WHY PRIMAL CAN'T USE KERNELS:                                               ║
║  ─────────────────────────────                                               ║
║  In Primal, we optimize w directly.                                         ║
║  To use kernels, we'd need to explicitly compute φ(x) for all samples,     ║
║  which could be infinite-dimensional!                                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHEN TO USE WHICH:                                                          ║
║  ──────────────────                                                          ║
║                                                                              ║
║  USE PRIMAL when:                                                            ║
║  • Linear kernel only                                                       ║
║  • n >> d (many samples, few features)                                      ║
║  • You need the explicit weight vector w                                    ║
║  • Example: Text classification with millions of documents                  ║
║                                                                              ║
║  USE DUAL when:                                                              ║
║  • You need non-linear kernels (RBF, Polynomial)                            ║
║  • d >> n (many features, few samples)                                      ║
║  • Example: Image classification, genomics                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================================
# PART 2: PRIMAL SVM IMPLEMENTATION (Linear Only)
# ============================================================================

class SVMPrimal:
    """
    SVM using PRIMAL formulation with Gradient Descent.
    
    This implementation directly optimizes the weight vector w.
    It can ONLY do linear classification (no kernel trick).
    
    Objective: min (1/2)||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))
    (This is the hinge loss formulation)
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        """
        Parameters:
        -----------
        C : float, regularization parameter
        learning_rate : float, step size for gradient descent
        n_iterations : int, number of training iterations
        """
        self.C = C
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = None  # Weight vector (what we're optimizing)
        self.b = 0     # Bias term
        
    def fit(self, X, y):
        """
        Train the SVM using gradient descent on the primal objective.
        
        The gradient of the objective function is:
        - For ||w||² term: w
        - For hinge loss: -yᵢxᵢ if yᵢ(w·xᵢ + b) < 1, else 0
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Convert labels to -1, 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Check if point is within margin or misclassified
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Point is correctly classified and outside margin
                    # Only regularization gradient
                    self.w -= self.lr * self.w
                else:
                    # Point is misclassified or within margin
                    # Regularization + hinge loss gradient
                    self.w -= self.lr * (self.w - self.C * y_[idx] * x_i)
                    self.b -= self.lr * (-self.C * y_[idx])
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
    def decision_function(self, X):
        """Return the distance to the hyperplane"""
        return np.dot(X, self.w) + self.b
    
    def score(self, X, y):
        """Calculate accuracy"""
        y_ = np.where(y <= 0, -1, 1)
        predictions = self.predict(X)
        return np.mean(predictions == y_)


# ============================================================================
# PART 3: DUAL SVM IMPLEMENTATION (Supports Kernels)
# ============================================================================

class SVMDual:
    """
    SVM using DUAL formulation with SMO algorithm.
    
    This implementation optimizes the Lagrange multipliers α.
    It CAN use the kernel trick for non-linear classification.
    
    Dual Objective: max Σαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
    
    Key insight: We never need w explicitly!
    Prediction: sign(Σᵢ αᵢyᵢK(xᵢ,x) + b)
    """
    
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3, 
                 coef0=1, tol=1e-3, max_iter=1000):
        """
        Parameters:
        -----------
        kernel : str, 'linear', 'rbf', or 'polynomial'
        C : float, regularization parameter (upper bound on α)
        gamma : float, kernel coefficient for RBF
        degree : int, degree for polynomial kernel
        coef0 : float, independent term for polynomial kernel
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        # These are what we optimize in DUAL
        self.alphas = None  # Lagrange multipliers
        self.b = 0
        
        # We need to store training data for prediction
        # (because prediction uses K(x_train, x_new))
        self.X_train = None
        self.y_train = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
    
    def _kernel_function(self, x1, x2):
        """
        Compute kernel between two samples.
        
        This is where the "magic" happens - we can compute
        dot products in high-dimensional spaces without
        explicitly transforming the data!
        """
        if self.kernel == 'linear':
            # K(x,y) = x·y
            # Equivalent to: φ(x)·φ(y) where φ(x) = x (identity)
            return np.dot(x1, x2)
        
        elif self.kernel == 'rbf':
            # K(x,y) = exp(-γ||x-y||²)
            # Equivalent to: φ(x)·φ(y) where φ maps to INFINITE dimensions!
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        
        elif self.kernel == 'polynomial':
            # K(x,y) = (x·y + coef0)^degree
            # Equivalent to: φ(x)·φ(y) where φ includes all polynomial terms
            return (np.dot(x1, x2) + self.coef0) ** self.degree
    
    def _compute_kernel_matrix(self, X):
        """
        Compute the kernel matrix K where K[i,j] = K(xᵢ, xⱼ).
        
        This matrix replaces the dot product matrix X @ X.T
        in the dual formulation.
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
                K[j, i] = K[i, j]  # Symmetric
        return K
    
    def fit(self, X, y):
        """
        Train using SMO (Sequential Minimal Optimization).
        
        SMO optimizes pairs of α values at a time, which allows
        us to solve analytically rather than using general QP solvers.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        n_samples = len(y)
        
        # Initialize all alphas to 0
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix ONCE (expensive but necessary)
        K = self._compute_kernel_matrix(self.X_train)
        
        # SMO algorithm
        for iteration in range(self.max_iter):
            alpha_changed = 0
            
            for i in range(n_samples):
                # Compute error for sample i
                Ei = self._compute_error(K, i)
                
                # Check KKT conditions
                if ((self.y_train[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (self.y_train[i] * Ei > self.tol and self.alphas[i] > 0)):
                    
                    # Select j
                    j = self._select_j(i, n_samples)
                    Ej = self._compute_error(K, j)
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta (second derivative of objective)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alphas[j] = alpha_j_old - self.y_train[j] * (Ei - Ej) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alphas[i] = alpha_i_old + self.y_train[i] * self.y_train[j] * \
                                     (alpha_j_old - self.alphas[j])
                    
                    # Update bias
                    b1 = self.b - Ei - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                         self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                         self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if alpha_changed == 0:
                break
        
        # Store support vectors (points where α > 0)
        sv_indices = self.alphas > 1e-8
        self.support_vector_indices_ = np.where(sv_indices)[0]
        self.support_vectors_ = self.X_train[sv_indices]
        
        return self
    
    def _compute_error(self, K, i):
        """Compute prediction error for sample i"""
        f_xi = np.sum(self.alphas * self.y_train * K[:, i]) + self.b
        return f_xi - self.y_train[i]
    
    def _select_j(self, i, n_samples):
        """Select second alpha to optimize"""
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def predict(self, X):
        """
        Predict class labels.
        
        Note: We don't use w·x + b
        Instead: sign(Σᵢ αᵢyᵢK(xᵢ,x) + b)
        
        This is the key difference from Primal!
        """
        X = np.array(X)
        predictions = []
        for x in X:
            # Sum over all support vectors
            decision = 0
            for i in range(len(self.X_train)):
                if self.alphas[i] > 1e-8:  # Only support vectors contribute
                    decision += self.alphas[i] * self.y_train[i] * \
                               self._kernel_function(self.X_train[i], x)
            decision += self.b
            predictions.append(1 if decision >= 0 else -1)
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# PART 4: COMPARISON DEMONSTRATION
# ============================================================================

def compare_implementations():
    """
    Compare Primal vs Dual on the same dataset
    """
    print("\n" + "=" * 70)
    print("PRIMAL vs DUAL IMPLEMENTATION COMPARISON")
    print("=" * 70)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    y = np.where(y == 0, -1, 1)  # Convert to -1, 1
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: Breast Cancer")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X.shape[1]}")
    
    # Use smaller subset for faster demo
    subset_size = 200
    indices = np.random.choice(len(y_train), subset_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]
    
    results = {}
    
    # 1. PRIMAL (Linear only)
    print("\n" + "-" * 70)
    print("PRIMAL FORMULATION (Linear SVM)")
    print("-" * 70)
    print("Optimizes: weight vector w directly")
    print("Variables: w ∈ ℝ^d (d = number of features)")
    print("Can use kernels: NO")
    
    svm_primal = SVMPrimal(C=1.0, learning_rate=0.001, n_iterations=1000)
    
    start_time = time.time()
    svm_primal.fit(X_train_sub, y_train_sub)
    primal_time = time.time() - start_time
    
    primal_train_acc = svm_primal.score(X_train_sub, y_train_sub)
    primal_test_acc = svm_primal.score(X_test, y_test)
    
    results['primal_linear'] = {
        'train_acc': primal_train_acc,
        'test_acc': primal_test_acc,
        'time': primal_time,
        'n_params': len(svm_primal.w) + 1  # w + b
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {primal_train_acc:.4f}")
    print(f"  Test Accuracy:  {primal_test_acc:.4f}")
    print(f"  Training Time:  {primal_time:.3f}s")
    print(f"  Parameters:     {len(svm_primal.w)} weights + 1 bias = {len(svm_primal.w) + 1}")
    print(f"  Weight vector:  w ∈ ℝ^{len(svm_primal.w)}")
    
    # 2. DUAL (Linear)
    print("\n" + "-" * 70)
    print("DUAL FORMULATION (Linear Kernel)")
    print("-" * 70)
    print("Optimizes: Lagrange multipliers α")
    print("Variables: α ∈ ℝ^n (n = number of samples)")
    print("Can use kernels: YES")
    
    svm_dual_linear = SVMDual(kernel='linear', C=1.0, max_iter=500)
    
    start_time = time.time()
    svm_dual_linear.fit(X_train_sub, y_train_sub)
    dual_linear_time = time.time() - start_time
    
    dual_linear_train_acc = svm_dual_linear.score(X_train_sub, y_train_sub)
    dual_linear_test_acc = svm_dual_linear.score(X_test, y_test)
    n_sv = len(svm_dual_linear.support_vectors_)
    
    results['dual_linear'] = {
        'train_acc': dual_linear_train_acc,
        'test_acc': dual_linear_test_acc,
        'time': dual_linear_time,
        'n_sv': n_sv
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy:    {dual_linear_train_acc:.4f}")
    print(f"  Test Accuracy:     {dual_linear_test_acc:.4f}")
    print(f"  Training Time:     {dual_linear_time:.3f}s")
    print(f"  Support Vectors:   {n_sv} out of {subset_size}")
    print(f"  Sparsity:          {100*(1 - n_sv/subset_size):.1f}% of αs are zero")
    
    # 3. DUAL (RBF Kernel)
    print("\n" + "-" * 70)
    print("DUAL FORMULATION (RBF Kernel)")
    print("-" * 70)
    print("Optimizes: Lagrange multipliers α")
    print("Kernel: K(x,y) = exp(-γ||x-y||²)")
    print("This is IMPOSSIBLE with Primal formulation!")
    
    svm_dual_rbf = SVMDual(kernel='rbf', C=10.0, gamma=0.01, max_iter=500)
    
    start_time = time.time()
    svm_dual_rbf.fit(X_train_sub, y_train_sub)
    dual_rbf_time = time.time() - start_time
    
    dual_rbf_train_acc = svm_dual_rbf.score(X_train_sub, y_train_sub)
    dual_rbf_test_acc = svm_dual_rbf.score(X_test, y_test)
    n_sv_rbf = len(svm_dual_rbf.support_vectors_)
    
    results['dual_rbf'] = {
        'train_acc': dual_rbf_train_acc,
        'test_acc': dual_rbf_test_acc,
        'time': dual_rbf_time,
        'n_sv': n_sv_rbf
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy:    {dual_rbf_train_acc:.4f}")
    print(f"  Test Accuracy:     {dual_rbf_test_acc:.4f}")
    print(f"  Training Time:     {dual_rbf_time:.3f}s")
    print(f"  Support Vectors:   {n_sv_rbf} out of {subset_size}")
    
    # 4. DUAL (Polynomial Kernel)
    print("\n" + "-" * 70)
    print("DUAL FORMULATION (Polynomial Kernel)")
    print("-" * 70)
    print("Optimizes: Lagrange multipliers α")
    print("Kernel: K(x,y) = (x·y + coef0)^degree")
    
    svm_dual_poly = SVMDual(kernel='polynomial', C=1.0, degree=3, coef0=1, max_iter=500)
    
    start_time = time.time()
    svm_dual_poly.fit(X_train_sub, y_train_sub)
    dual_poly_time = time.time() - start_time
    
    dual_poly_train_acc = svm_dual_poly.score(X_train_sub, y_train_sub)
    dual_poly_test_acc = svm_dual_poly.score(X_test, y_test)
    n_sv_poly = len(svm_dual_poly.support_vectors_)
    
    results['dual_poly'] = {
        'train_acc': dual_poly_train_acc,
        'test_acc': dual_poly_test_acc,
        'time': dual_poly_time,
        'n_sv': n_sv_poly
    }
    
    print(f"\nResults:")
    print(f"  Train Accuracy:    {dual_poly_train_acc:.4f}")
    print(f"  Test Accuracy:     {dual_poly_test_acc:.4f}")
    print(f"  Training Time:     {dual_poly_time:.3f}s")
    print(f"  Support Vectors:   {n_sv_poly} out of {subset_size}")
    
    # Summary Table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<25} {'Test Acc':<12} {'Train Time':<12} {'Notes'}")
    print("-" * 70)
    print(f"{'Primal (Linear)':<25} {primal_test_acc:.4f}       {primal_time:.3f}s        w ∈ ℝ^{X.shape[1]}")
    print(f"{'Dual (Linear)':<25} {dual_linear_test_acc:.4f}       {dual_linear_time:.3f}s        {n_sv} SVs")
    print(f"{'Dual (RBF)':<25} {dual_rbf_test_acc:.4f}       {dual_rbf_time:.3f}s        {n_sv_rbf} SVs")
    print(f"{'Dual (Polynomial)':<25} {dual_poly_test_acc:.4f}       {dual_poly_time:.3f}s        {n_sv_poly} SVs")
    
    return results


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def visualize_primal_vs_dual():
    """
    Create visualization showing the difference between Primal and Dual
    """
    # Create simple 2D dataset for visualization
    np.random.seed(42)
    
    # Generate non-linearly separable data (circles)
    n_samples = 200
    
    # Inner circle (class -1)
    r1 = np.random.uniform(0, 1, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.ones(n_samples // 2) * -1
    
    # Outer ring (class 1)
    r2 = np.random.uniform(1.5, 2.5, n_samples // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = np.ones(n_samples // 2)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Linear SVM (Primal - will fail)
    ax = axes[0]
    svm_linear = SVMPrimal(C=1.0, learning_rate=0.01, n_iterations=1000)
    svm_linear.fit(X, y)
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='o', s=30, label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', s=30, label='Class +1')
    ax.set_title(f'PRIMAL (Linear)\nAccuracy: {svm_linear.score(X, y):.2%}\n(Cannot separate!)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    
    # 2. Linear SVM (Dual - also fails but shows the approach)
    ax = axes[1]
    svm_dual_linear = SVMDual(kernel='linear', C=1.0, max_iter=500)
    svm_dual_linear.fit(X, y)
    
    Z = svm_dual_linear.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='o', s=30, label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', s=30, label='Class +1')
    ax.set_title(f'DUAL (Linear Kernel)\nAccuracy: {svm_dual_linear.score(X, y):.2%}\n(Still linear!)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    
    # 3. RBF SVM (Dual - succeeds!)
    ax = axes[2]
    svm_dual_rbf = SVMDual(kernel='rbf', C=10.0, gamma=1.0, max_iter=500)
    svm_dual_rbf.fit(X, y)
    
    Z = svm_dual_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='o', s=30, label='Class -1')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', s=30, label='Class +1')
    
    # Highlight support vectors
    sv_indices = svm_dual_rbf.support_vector_indices_
    ax.scatter(X[sv_indices, 0], X[sv_indices, 1], s=100, linewidth=2,
               facecolors='none', edgecolors='green', label='Support Vectors')
    
    ax.set_title(f'DUAL (RBF Kernel)\nAccuracy: {svm_dual_rbf.score(X, y):.2%}\n(Kernel Trick works!)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    
    plt.suptitle('Why We Need DUAL Formulation for Kernels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('primal_vs_dual_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: primal_vs_dual_visualization.png")


# ============================================================================
# PART 6: WHY I USED DUAL IN THE ORIGINAL CODE
# ============================================================================

def explain_why_dual():
    """
    Explain why Dual was used in the original implementation
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              WHY I USED DUAL IN THE ORIGINAL IMPLEMENTATION                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE ORIGINAL REQUEST:                                                       ║
║  ─────────────────────                                                       ║
║  "Implement Linear Kernel, RBF Kernel, and Polynomial Kernel"               ║
║                                                                              ║
║  THE KEY CONSTRAINT:                                                         ║
║  ───────────────────                                                         ║
║  • RBF and Polynomial kernels REQUIRE the Dual formulation                  ║
║  • The kernel trick ONLY works in Dual form                                 ║
║  • Primal can only do linear SVM (no kernels)                               ║
║                                                                              ║
║  THEREFORE:                                                                  ║
║  ──────────                                                                  ║
║  To support ALL THREE kernel types with a UNIFIED implementation,           ║
║  the Dual formulation was the ONLY viable choice.                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MATHEMATICAL REASON:                                                        ║
║  ────────────────────                                                        ║
║                                                                              ║
║  PRIMAL prediction:  f(x) = w · x + b                                        ║
║                                                                              ║
║  To use kernel:      f(x) = w · φ(x) + b                                    ║
║                                                                              ║
║  Problem: For RBF, φ(x) is INFINITE dimensional!                            ║
║           We cannot compute or store w ∈ ℝ^∞                                ║
║                                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  DUAL prediction:    f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b                          ║
║                                                                              ║
║  We never need φ(x)! We only need K(xᵢ, x) = φ(xᵢ) · φ(x)                  ║
║  The kernel function computes this dot product IMPLICITLY.                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SUMMARY TABLE:                                                              ║
║  ──────────────                                                              ║
║                                                                              ║
║  ┌─────────────────┬────────────────────┬────────────────────┐              ║
║  │   Kernel        │   Primal           │   Dual             │              ║
║  ├─────────────────┼────────────────────┼────────────────────┤              ║
║  │   Linear        │   ✓ Works          │   ✓ Works          │              ║
║  │   RBF           │   ✗ Impossible     │   ✓ Works          │              ║
║  │   Polynomial    │   ✗ Impractical    │   ✓ Works          │              ║
║  └─────────────────┴────────────────────┴────────────────────┘              ║
║                                                                              ║
║  For a unified implementation supporting all kernels → MUST use Dual        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   SVM: PRIMAL vs DUAL FORMULATION - COMPLETE GUIDE")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Theoretical explanation
    explain_primal_vs_dual()
    
    # Practical comparison
    results = compare_implementations()
    
    # Visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    visualize_primal_vs_dual()
    
    # Explain the choice
    explain_why_dual()
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE!")
    print("=" * 80)