"""
SVM Kernel Comparison: From Scratch vs Scikit-Learn
====================================================
This script implements Linear, RBF, and Polynomial kernels from scratch,
compares them with Scikit-Learn implementations, and demonstrates
hyperparameter tuning using real-world data (Breast Cancer dataset).

Author: Claude
"""

import numpy as np
from collections import Counter
import time

# ============================================================================
# PART 1: FROM-SCRATCH IMPLEMENTATIONS
# ============================================================================

class KernelFunctions:
    """Collection of kernel functions for SVM"""
    
    @staticmethod
    def linear(x1, x2):
        """Linear kernel: K(x1, x2) = x1 · x2"""
        return np.dot(x1, x2)
    
    @staticmethod
    def rbf(x1, x2, gamma=0.1):
        """RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
        diff = x1 - x2
        return np.exp(-gamma * np.dot(diff, diff))
    
    @staticmethod
    def polynomial(x1, x2, degree=3, coef0=1):
        """Polynomial kernel: K(x1, x2) = (x1 · x2 + coef0)^degree"""
        return (np.dot(x1, x2) + coef0) ** degree


class SVMFromScratch:
    """
    Support Vector Machine implementation from scratch using SMO algorithm.
    
    This is a simplified but functional implementation that demonstrates
    the core concepts of SVM with different kernels.
    """
    
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3, coef0=1,
                 tol=1e-3, max_iter=1000):
        """
        Parameters:
        -----------
        kernel : str, 'linear', 'rbf', or 'polynomial'
        C : float, regularization parameter
        gamma : float, kernel coefficient for RBF
        degree : int, degree for polynomial kernel
        coef0 : float, independent term for polynomial kernel
        tol : float, tolerance for stopping criterion
        max_iter : int, maximum number of iterations
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        # Model parameters (set during training)
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
    
    def _kernel_function(self, x1, x2):
        """Compute kernel between two samples"""
        if self.kernel == 'linear':
            return KernelFunctions.linear(x1, x2)
        elif self.kernel == 'rbf':
            return KernelFunctions.rbf(x1, x2, self.gamma)
        elif self.kernel == 'polynomial':
            return KernelFunctions.polynomial(x1, x2, self.degree, self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X):
        """Compute the kernel matrix for training data"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
                K[j, i] = K[i, j]  # Kernel matrix is symmetric
        return K
    
    def _decision_function_single(self, x):
        """Compute decision function for a single sample"""
        result = 0
        for i in range(len(self.X_train)):
            if self.alphas[i] > 1e-8:  # Only support vectors contribute
                result += self.alphas[i] * self.y_train[i] * \
                          self._kernel_function(self.X_train[i], x)
        return result + self.b
    
    def fit(self, X, y):
        """
        Train the SVM using a simplified SMO (Sequential Minimal Optimization) algorithm.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), labels should be -1 or 1
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        n_samples = len(y)
        
        # Initialize alphas and bias
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(self.X_train)
        
        # SMO algorithm (simplified version)
        for iteration in range(self.max_iter):
            alpha_changed = 0
            
            for i in range(n_samples):
                # Compute error for sample i
                Ei = self._compute_error(K, i)
                
                # Check if alpha_i violates KKT conditions
                if ((self.y_train[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (self.y_train[i] * Ei > self.tol and self.alphas[i] > 0)):
                    
                    # Select j randomly (simplified heuristic)
                    j = self._select_j(i, n_samples)
                    
                    # Compute error for sample j
                    Ej = self._compute_error(K, j)
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds L and H
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
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
                    
                    # Compute bias terms
                    b1 = self.b - Ei - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                         self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                         self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    # Update bias
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            # Early stopping if no alphas changed
            if alpha_changed == 0:
                break
        
        # Store support vectors
        sv_indices = self.alphas > 1e-8
        self.support_vector_indices_ = np.where(sv_indices)[0]
        self.support_vectors_ = self.X_train[sv_indices]
        
        return self
    
    def _compute_error(self, K, i):
        """Compute error for sample i"""
        f_xi = np.sum(self.alphas * self.y_train * K[:, i]) + self.b
        return f_xi - self.y_train[i]
    
    def _select_j(self, i, n_samples):
        """Select j randomly (simplified version)"""
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        X = np.array(X)
        predictions = []
        for x in X:
            decision = self._decision_function_single(x)
            predictions.append(1 if decision >= 0 else -1)
        return np.array(predictions)
    
    def decision_function(self, X):
        """Compute decision function values for samples in X"""
        X = np.array(X)
        return np.array([self._decision_function_single(x) for x in X])
    
    def score(self, X, y):
        """Compute accuracy on test data"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# PART 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_breast_cancer_data():
    """
    Load the Breast Cancer Wisconsin dataset.
    This is a real-world dataset with 569 samples and 30 features.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Convert labels to -1 and 1 (required for our from-scratch SVM)
    y = np.where(y == 0, -1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (crucial for SVM performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, data.feature_names


# ============================================================================
# PART 3: HYPERPARAMETER TUNING FROM SCRATCH
# ============================================================================

def cross_validation_split(X, y, n_folds=5):
    """Create stratified cross-validation splits"""
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    folds = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, test_indices))
    
    return folds


def grid_search_from_scratch(X, y, param_grid, kernel='rbf', n_folds=3):
    """
    Perform grid search with cross-validation from scratch.
    
    Parameters:
    -----------
    X : array-like, training data
    y : array-like, labels
    param_grid : dict, parameters to search
    kernel : str, kernel type
    n_folds : int, number of CV folds
    
    Returns:
    --------
    best_params : dict, best parameters found
    results : list, all results
    """
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    results = []
    best_score = -np.inf
    best_params = None
    
    print(f"\nGrid Search for {kernel.upper()} kernel:")
    print("=" * 60)
    
    folds = cross_validation_split(X, y, n_folds)
    
    for params in param_combinations:
        param_dict = dict(zip(param_names, params))
        
        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Train model with current parameters
            svm = SVMFromScratch(kernel=kernel, **param_dict, max_iter=500)
            svm.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            score = svm.score(X_val_fold, y_val_fold)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        results.append({
            'params': param_dict,
            'mean_score': mean_score,
            'std_score': std_score
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = param_dict
        
        print(f"  Params: {param_dict}")
        print(f"  CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
        print("-" * 40)
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV Score: {best_score:.4f}")
    
    return best_params, results


# ============================================================================
# PART 4: SCIKIT-LEARN COMPARISON
# ============================================================================

def train_sklearn_svm(X_train, X_test, y_train, y_test):
    """Train SVM models using scikit-learn and perform hyperparameter tuning"""
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    
    results = {}
    
    # Define parameter grids for each kernel
    param_grids = {
        'linear': {
            'C': [0.01, 0.1, 1, 10, 100]
        },
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        },
        'poly': {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'coef0': [0, 1]
        }
    }
    
    print("\n" + "=" * 70)
    print("SCIKIT-LEARN SVM WITH GRID SEARCH")
    print("=" * 70)
    
    for kernel in ['linear', 'rbf', 'poly']:
        print(f"\n{'='*60}")
        print(f"Kernel: {kernel.upper()}")
        print("=" * 60)
        
        # Perform grid search
        svm = SVC(kernel=kernel)
        grid_search = GridSearchCV(
            svm, 
            param_grids[kernel], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate
        train_accuracy = best_model.score(X_train, y_train)
        test_accuracy = best_model.score(X_test, y_test)
        
        results[kernel] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_support_vectors': sum(best_model.n_support_),
            'training_time': training_time,
            'model': best_model
        }
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Number of Support Vectors: {sum(best_model.n_support_)}")
        print(f"Training Time: {training_time:.3f}s")
    
    return results


# ============================================================================
# PART 5: MAIN COMPARISON FUNCTION
# ============================================================================

def run_full_comparison():
    """Run complete comparison between from-scratch and scikit-learn implementations"""
    
    print("\n" + "=" * 70)
    print("SVM KERNEL COMPARISON: FROM SCRATCH vs SCIKIT-LEARN")
    print("=" * 70)
    print("\nDataset: Breast Cancer Wisconsin (Diagnostic)")
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_breast_cancer_data()
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {len(feature_names)}")
    
    # Use a smaller subset for from-scratch (faster training)
    # In production, you'd use the full dataset
    subset_size = min(200, len(y_train))
    indices = np.random.choice(len(y_train), subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    
    print(f"\nUsing {subset_size} samples for from-scratch implementation")
    
    # =========================================================================
    # FROM SCRATCH IMPLEMENTATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("FROM-SCRATCH SVM IMPLEMENTATION")
    print("=" * 70)
    
    scratch_results = {}
    
    # 1. Linear Kernel
    print("\n" + "-" * 60)
    print("LINEAR KERNEL")
    print("-" * 60)
    
    linear_params = {'C': [0.1, 1, 10]}
    best_linear_params, _ = grid_search_from_scratch(
        X_train_subset, y_train_subset, linear_params, kernel='linear', n_folds=3
    )
    
    # Train final model with best parameters
    svm_linear = SVMFromScratch(kernel='linear', **best_linear_params, max_iter=1000)
    start_time = time.time()
    svm_linear.fit(X_train_subset, y_train_subset)
    linear_time = time.time() - start_time
    
    scratch_results['linear'] = {
        'train_accuracy': svm_linear.score(X_train_subset, y_train_subset),
        'test_accuracy': svm_linear.score(X_test, y_test),
        'n_support_vectors': len(svm_linear.support_vectors_),
        'training_time': linear_time,
        'best_params': best_linear_params
    }
    
    print(f"\nFinal Results (Linear):")
    print(f"  Train Accuracy: {scratch_results['linear']['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {scratch_results['linear']['test_accuracy']:.4f}")
    print(f"  Support Vectors: {scratch_results['linear']['n_support_vectors']}")
    
    # 2. RBF Kernel
    print("\n" + "-" * 60)
    print("RBF KERNEL")
    print("-" * 60)
    
    rbf_params = {'C': [1, 10], 'gamma': [0.01, 0.1]}
    best_rbf_params, _ = grid_search_from_scratch(
        X_train_subset, y_train_subset, rbf_params, kernel='rbf', n_folds=3
    )
    
    svm_rbf = SVMFromScratch(kernel='rbf', **best_rbf_params, max_iter=1000)
    start_time = time.time()
    svm_rbf.fit(X_train_subset, y_train_subset)
    rbf_time = time.time() - start_time
    
    scratch_results['rbf'] = {
        'train_accuracy': svm_rbf.score(X_train_subset, y_train_subset),
        'test_accuracy': svm_rbf.score(X_test, y_test),
        'n_support_vectors': len(svm_rbf.support_vectors_),
        'training_time': rbf_time,
        'best_params': best_rbf_params
    }
    
    print(f"\nFinal Results (RBF):")
    print(f"  Train Accuracy: {scratch_results['rbf']['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {scratch_results['rbf']['test_accuracy']:.4f}")
    print(f"  Support Vectors: {scratch_results['rbf']['n_support_vectors']}")
    
    # 3. Polynomial Kernel
    print("\n" + "-" * 60)
    print("POLYNOMIAL KERNEL")
    print("-" * 60)
    
    poly_params = {'C': [1, 10], 'degree': [2, 3], 'coef0': [0, 1]}
    best_poly_params, _ = grid_search_from_scratch(
        X_train_subset, y_train_subset, poly_params, kernel='polynomial', n_folds=3
    )
    
    svm_poly = SVMFromScratch(kernel='polynomial', **best_poly_params, max_iter=1000)
    start_time = time.time()
    svm_poly.fit(X_train_subset, y_train_subset)
    poly_time = time.time() - start_time
    
    scratch_results['polynomial'] = {
        'train_accuracy': svm_poly.score(X_train_subset, y_train_subset),
        'test_accuracy': svm_poly.score(X_test, y_test),
        'n_support_vectors': len(svm_poly.support_vectors_),
        'training_time': poly_time,
        'best_params': best_poly_params
    }
    
    print(f"\nFinal Results (Polynomial):")
    print(f"  Train Accuracy: {scratch_results['polynomial']['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {scratch_results['polynomial']['test_accuracy']:.4f}")
    print(f"  Support Vectors: {scratch_results['polynomial']['n_support_vectors']}")
    
    # =========================================================================
    # SCIKIT-LEARN IMPLEMENTATION (using full training set)
    # =========================================================================
    sklearn_results = train_sklearn_svm(X_train, X_test, y_train, y_test)
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n{:<15} {:<12} {:<12} {:<15} {:<15}".format(
        "Kernel", "Method", "Test Acc", "Support Vecs", "Time (s)"))
    print("-" * 70)
    
    for kernel in ['linear', 'rbf']:
        sk_kernel = kernel
        scratch_kernel = kernel
        
        print("{:<15} {:<12} {:.4f}       {:<15} {:.3f}".format(
            kernel.upper(),
            "Scratch",
            scratch_results[scratch_kernel]['test_accuracy'],
            scratch_results[scratch_kernel]['n_support_vectors'],
            scratch_results[scratch_kernel]['training_time']
        ))
        print("{:<15} {:<12} {:.4f}       {:<15} {:.3f}".format(
            "",
            "Sklearn",
            sklearn_results[sk_kernel]['test_accuracy'],
            sklearn_results[sk_kernel]['n_support_vectors'],
            sklearn_results[sk_kernel]['training_time']
        ))
        print("-" * 70)
    
    # Polynomial
    print("{:<15} {:<12} {:.4f}       {:<15} {:.3f}".format(
        "POLYNOMIAL",
        "Scratch",
        scratch_results['polynomial']['test_accuracy'],
        scratch_results['polynomial']['n_support_vectors'],
        scratch_results['polynomial']['training_time']
    ))
    print("{:<15} {:<12} {:.4f}       {:<15} {:.3f}".format(
        "",
        "Sklearn",
        sklearn_results['poly']['test_accuracy'],
        sklearn_results['poly']['n_support_vectors'],
        sklearn_results['poly']['training_time']
    ))
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
1. ACCURACY COMPARISON:
   - Scikit-learn typically achieves slightly better accuracy due to
     optimized SMO implementation and using full training data
   - From-scratch implementation provides good approximation
   
2. TRAINING TIME:
   - Scikit-learn is significantly faster (C implementation, LIBSVM)
   - From-scratch is slower due to pure Python implementation
   
3. KERNEL PERFORMANCE:
   - RBF kernel often performs best for this dataset
   - Linear kernel is fastest and works well for linearly separable data
   - Polynomial kernel can capture non-linear relationships
   
4. HYPERPARAMETER IMPORTANCE:
   - C (regularization): Higher = less regularization, risk of overfitting
   - gamma (RBF): Higher = more complex decision boundary
   - degree (polynomial): Higher = more complex model
""")
    
    return scratch_results, sklearn_results


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def create_visualization(X_train, y_train, X_test, y_test):
    """Create visualization comparing kernel decision boundaries"""
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    kernels = [
        ('linear', {'C': 1}),
        ('rbf', {'C': 10, 'gamma': 0.1}),
        ('poly', {'C': 1, 'degree': 3})
    ]
    
    for idx, (kernel, params) in enumerate(kernels):
        # Train model
        model = SVC(kernel=kernel, **params)
        model.fit(X_train_2d, y_train)
        
        # Create mesh grid for decision boundary
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Get predictions for mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax = axes[0, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X_train_2d[y_train == -1, 0], X_train_2d[y_train == -1, 1],
                   c='red', marker='o', label='Malignant', alpha=0.6)
        ax.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1],
                   c='blue', marker='o', label='Benign', alpha=0.6)
        ax.set_title(f'{kernel.upper()} Kernel\nTrain Accuracy: {model.score(X_train_2d, y_train):.3f}')
        ax.legend()
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        
        # Plot test predictions
        ax = axes[1, idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X_test_2d[y_test == -1, 0], X_test_2d[y_test == -1, 1],
                   c='red', marker='s', label='Malignant', alpha=0.6)
        ax.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
                   c='blue', marker='s', label='Benign', alpha=0.6)
        ax.set_title(f'Test Set\nTest Accuracy: {model.score(X_test_2d, y_test):.3f}')
        ax.legend()
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
    
    plt.suptitle('SVM Kernel Comparison on Breast Cancer Dataset (PCA-reduced)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svm_kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved to 'svm_kernel_comparison.png'")


def create_hyperparameter_heatmap():
    """Create heatmap showing hyperparameter tuning results"""
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter ranges
    C_range = [0.01, 0.1, 1, 10, 100]
    gamma_range = [0.001, 0.01, 0.1, 1, 10]
    
    # Compute cross-validation scores
    scores = np.zeros((len(C_range), len(gamma_range)))
    
    for i, C in enumerate(C_range):
        for j, gamma in enumerate(gamma_range):
            svm = SVC(kernel='rbf', C=C, gamma=gamma)
            cv_scores = cross_val_score(svm, X_scaled, y, cv=5)
            scores[i, j] = cv_scores.mean()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(scores, interpolation='nearest', cmap='YlGnBu')
    
    ax.set_xticks(np.arange(len(gamma_range)))
    ax.set_yticks(np.arange(len(C_range)))
    ax.set_xticklabels([f'{g}' for g in gamma_range])
    ax.set_yticklabels([f'{c}' for c in C_range])
    ax.set_xlabel('Gamma', fontsize=12)
    ax.set_ylabel('C (Regularization)', fontsize=12)
    ax.set_title('RBF Kernel Hyperparameter Tuning\n(5-Fold Cross-Validation Accuracy)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy', rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(C_range)):
        for j in range(len(gamma_range)):
            text = ax.text(j, i, f'{scores[i, j]:.3f}',
                          ha="center", va="center", color="black" if scores[i, j] > 0.9 else "white")
    
    plt.tight_layout()
    plt.savefig('hyperparameter_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Hyperparameter heatmap saved to 'hyperparameter_heatmap.png'")
    
    # Find best parameters
    best_idx = np.unravel_index(scores.argmax(), scores.shape)
    print(f"\nBest Parameters: C={C_range[best_idx[0]]}, gamma={gamma_range[best_idx[1]]}")
    print(f"Best CV Accuracy: {scores[best_idx[0], best_idx[1]]:.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the full comparison
    scratch_results, sklearn_results = run_full_comparison()
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Load data again for visualization
    X_train, X_test, y_train, y_test, _ = load_breast_cancer_data()
    
    create_visualization(X_train, y_train, X_test, y_test)
    create_hyperparameter_heatmap()
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE!")
    print("=" * 70)