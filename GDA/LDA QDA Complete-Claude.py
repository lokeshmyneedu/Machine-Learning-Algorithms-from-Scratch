"""
Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
Complete Implementation from Scratch with:
- Feature Selection
- Regularization
- Feature Standardization
- Real-world Data (Wine Dataset)
- Scikit-learn Comparison
- PyTorch Implementation
- TensorFlow Implementation

Author: Claude
"""

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: FEATURE STANDARDIZATION CLASS (FROM SCRATCH)
# ============================================================================

class StandardScalerFromScratch:
    """
    Feature standardization: z = (x - mean) / std
    Transforms features to have zero mean and unit variance.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
        
    def fit(self, X):
        """Compute mean and std from training data."""
        X = np.array(X)
        self.n_features_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)
        # Handle zero variance features
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Apply standardization."""
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """Reverse the standardization."""
        return X_scaled * self.std_ + self.mean_


# ============================================================================
# PART 2: FEATURE SELECTION CLASS (FROM SCRATCH)
# ============================================================================

class FeatureSelectorFromScratch:
    """
    Feature selection using various criteria:
    - F-statistic (ANOVA)
    - Fisher's criterion (for LDA)
    - Mutual information approximation
    """
    
    def __init__(self, method='fisher', k=None):
        """
        Parameters:
        -----------
        method : str
            'fisher' - Fisher's discriminant ratio
            'f_statistic' - ANOVA F-statistic
            'variance' - Variance-based selection
        k : int
            Number of features to select (None = all)
        """
        self.method = method
        self.k = k
        self.scores_ = None
        self.selected_features_ = None
        
    def _compute_fisher_score(self, X, y):
        """
        Fisher's criterion: (between-class variance) / (within-class variance)
        """
        classes = np.unique(y)
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        
        overall_mean = np.mean(X, axis=0)
        
        for j in range(n_features):
            # Between-class variance
            between_var = 0
            within_var = 0
            
            for c in classes:
                X_c = X[y == c, j]
                n_c = len(X_c)
                class_mean = np.mean(X_c)
                class_var = np.var(X_c)
                
                between_var += n_c * (class_mean - overall_mean[j]) ** 2
                within_var += n_c * class_var
            
            # Avoid division by zero
            if within_var > 1e-10:
                scores[j] = between_var / within_var
            else:
                scores[j] = 0
                
        return scores
    
    def _compute_f_statistic(self, X, y):
        """
        ANOVA F-statistic for each feature.
        """
        classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(classes)
        
        scores = np.zeros(n_features)
        overall_mean = np.mean(X, axis=0)
        
        for j in range(n_features):
            # Sum of squares between groups
            ss_between = 0
            # Sum of squares within groups
            ss_within = 0
            
            for c in classes:
                X_c = X[y == c, j]
                n_c = len(X_c)
                class_mean = np.mean(X_c)
                
                ss_between += n_c * (class_mean - overall_mean[j]) ** 2
                ss_within += np.sum((X_c - class_mean) ** 2)
            
            # Degrees of freedom
            df_between = n_classes - 1
            df_within = n_samples - n_classes
            
            # F-statistic
            if ss_within > 1e-10 and df_within > 0:
                ms_between = ss_between / df_between
                ms_within = ss_within / df_within
                scores[j] = ms_between / ms_within
            else:
                scores[j] = 0
                
        return scores
    
    def _compute_variance_score(self, X, y):
        """Variance-based feature importance."""
        return np.var(X, axis=0)
    
    def fit(self, X, y):
        """Compute feature scores and select top k."""
        X = np.array(X)
        y = np.array(y)
        
        if self.method == 'fisher':
            self.scores_ = self._compute_fisher_score(X, y)
        elif self.method == 'f_statistic':
            self.scores_ = self._compute_f_statistic(X, y)
        elif self.method == 'variance':
            self.scores_ = self._compute_variance_score(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Select top k features
        if self.k is None or self.k >= X.shape[1]:
            self.selected_features_ = np.arange(X.shape[1])
        else:
            self.selected_features_ = np.argsort(self.scores_)[::-1][:self.k]
            
        return self
    
    def transform(self, X):
        """Select the top k features."""
        X = np.array(X)
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


# ============================================================================
# PART 3: LINEAR DISCRIMINANT ANALYSIS FROM SCRATCH
# ============================================================================

class LDAFromScratch:
    """
    Linear Discriminant Analysis implementation from scratch.
    
    Supports:
    - Multi-class classification
    - Regularization (shrinkage)
    - Dimensionality reduction
    """
    
    def __init__(self, n_components=None, shrinkage=None, solver='svd'):
        """
        Parameters:
        -----------
        n_components : int
            Number of components for dimensionality reduction
        shrinkage : float or 'auto'
            Regularization parameter (0 to 1)
            - None: No regularization
            - 'auto': Ledoit-Wolf shrinkage
            - float: Manual shrinkage coefficient
        solver : str
            'svd' or 'eigen'
        """
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.solver = solver
        
        # Fitted parameters
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.scalings_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def _compute_class_statistics(self, X, y):
        """Compute class means and priors."""
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Class priors
        self.priors_ = np.zeros(n_classes)
        # Class means
        self.means_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = len(X_c) / n_samples
            self.means_[i] = np.mean(X_c, axis=0)
            
    def _compute_within_class_covariance(self, X, y):
        """Compute pooled within-class covariance matrix."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Initialize covariance
        Sw = np.zeros((n_features, n_features))
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            X_centered = X_c - self.means_[i]
            Sw += X_centered.T @ X_centered
            
        # Normalize
        Sw /= (n_samples - n_classes)
        
        return Sw
    
    def _apply_shrinkage(self, cov, n_samples):
        """Apply shrinkage regularization to covariance matrix."""
        n_features = cov.shape[0]
        
        if self.shrinkage is None:
            return cov
        elif self.shrinkage == 'auto':
            # Ledoit-Wolf shrinkage estimation
            shrinkage = self._ledoit_wolf_shrinkage(cov, n_samples)
        else:
            shrinkage = self.shrinkage
            
        # Shrunk covariance: (1-s)*Sigma + s*trace(Sigma)/p * I
        trace = np.trace(cov)
        identity = np.eye(n_features)
        return (1 - shrinkage) * cov + shrinkage * (trace / n_features) * identity
    
    def _ledoit_wolf_shrinkage(self, cov, n_samples):
        """Estimate optimal shrinkage using Ledoit-Wolf formula."""
        n_features = cov.shape[0]
        
        # Simple approximation
        trace = np.trace(cov)
        trace_sq = np.trace(cov @ cov)
        
        # Ledoit-Wolf shrinkage
        mu = trace / n_features
        delta = (trace_sq + trace**2) / (n_features * (n_features + 1))
        
        # Estimate shrinkage (simplified)
        shrinkage = min(1.0, max(0.0, (delta - mu**2) / (n_samples * delta)))
        
        return shrinkage
    
    def _compute_between_class_covariance(self, X, y):
        """Compute between-class covariance matrix."""
        n_features = X.shape[1]
        overall_mean = np.mean(X, axis=0)
        
        Sb = np.zeros((n_features, n_features))
        
        for i, c in enumerate(self.classes_):
            n_c = np.sum(y == c)
            mean_diff = (self.means_[i] - overall_mean).reshape(-1, 1)
            Sb += n_c * (mean_diff @ mean_diff.T)
            
        return Sb
    
    def fit(self, X, y):
        """Fit LDA model."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Compute class statistics
        self._compute_class_statistics(X, y)
        n_classes = len(self.classes_)
        
        # Compute covariance matrices
        Sw = self._compute_within_class_covariance(X, y)
        Sb = self._compute_between_class_covariance(X, y)
        
        # Apply regularization
        Sw = self._apply_shrinkage(Sw, n_samples)
        self.covariance_ = Sw
        
        # Solve generalized eigenvalue problem
        if self.solver == 'svd':
            # Use SVD for numerical stability
            U, S, Vt = np.linalg.svd(Sw)
            # Pseudo-inverse
            S_inv = np.zeros_like(S)
            S_inv[S > 1e-10] = 1.0 / S[S > 1e-10]
            Sw_inv = Vt.T @ np.diag(S_inv) @ U.T
            
            # Eigendecomposition of Sw^-1 @ Sb
            eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
        else:
            # Direct eigendecomposition
            eigvals, eigvecs = linalg.eigh(Sb, Sw)
            
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Keep only real parts
        eigvecs = np.real(eigvecs)
        
        # Number of discriminant components
        max_components = min(n_features, n_classes - 1)
        if self.n_components is None:
            self.n_components = max_components
        else:
            self.n_components = min(self.n_components, max_components)
            
        self.scalings_ = eigvecs[:, :self.n_components]
        
        # Compute coefficients for classification
        self._compute_classification_params()
        
        return self
    
    def _compute_classification_params(self):
        """Compute linear coefficients and intercepts."""
        n_classes = len(self.classes_)
        n_features = self.covariance_.shape[0]
        
        # Inverse of pooled covariance
        try:
            cov_inv = np.linalg.inv(self.covariance_)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(self.covariance_)
        
        # For binary classification: w = Sigma^-1 (mu1 - mu0)
        # For multi-class: compute discriminant functions
        self.coef_ = self.means_ @ cov_inv
        self.intercept_ = -0.5 * np.sum(self.means_ * (self.means_ @ cov_inv), axis=1)
        self.intercept_ += np.log(self.priors_)
        
    def transform(self, X):
        """Project data to discriminant subspace."""
        X = np.array(X, dtype=np.float64)
        return X @ self.scalings_
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X, dtype=np.float64)
        
        # Discriminant scores
        scores = X @ self.coef_.T + self.intercept_
        
        # Softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


# ============================================================================
# PART 4: QUADRATIC DISCRIMINANT ANALYSIS FROM SCRATCH
# ============================================================================

class QDAFromScratch:
    """
    Quadratic Discriminant Analysis implementation from scratch.
    
    Supports:
    - Multi-class classification
    - Regularization (shrinkage)
    - Individual class covariances
    """
    
    def __init__(self, reg_param=0.0, store_covariance=True):
        """
        Parameters:
        -----------
        reg_param : float
            Regularization parameter (0 to 1)
            Shrinks covariances towards diagonal
        store_covariance : bool
            Whether to store covariance matrices
        """
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        
        # Fitted parameters
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariances_ = None
        self.covariance_det_ = None
        self.covariance_inv_ = None
        
    def _compute_class_statistics(self, X, y):
        """Compute class-specific statistics."""
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.covariances_ = []
        self.covariance_det_ = np.zeros(n_classes)
        self.covariance_inv_ = []
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = len(X_c)
            
            # Prior
            self.priors_[i] = n_c / n_samples
            
            # Mean
            self.means_[i] = np.mean(X_c, axis=0)
            
            # Covariance
            X_centered = X_c - self.means_[i]
            cov = (X_centered.T @ X_centered) / (n_c - 1)
            
            # Regularization
            if self.reg_param > 0:
                cov = self._apply_regularization(cov)
                
            if self.store_covariance:
                self.covariances_.append(cov)
            
            # Compute determinant and inverse for prediction
            try:
                self.covariance_det_[i] = np.linalg.det(cov)
                self.covariance_inv_.append(np.linalg.inv(cov))
            except np.linalg.LinAlgError:
                # Use pseudo-inverse for singular matrices
                self.covariance_det_[i] = np.prod(np.linalg.svd(cov, compute_uv=False))
                self.covariance_inv_.append(np.linalg.pinv(cov))
                
    def _apply_regularization(self, cov):
        """Apply shrinkage regularization."""
        n_features = cov.shape[0]
        # Shrink towards diagonal
        trace = np.trace(cov) / n_features
        return (1 - self.reg_param) * cov + self.reg_param * trace * np.eye(n_features)
    
    def fit(self, X, y):
        """Fit QDA model."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        self._compute_class_statistics(X, y)
        return self
    
    def _compute_discriminant_function(self, X, class_idx):
        """Compute quadratic discriminant function for a class."""
        mean = self.means_[class_idx]
        cov_inv = self.covariance_inv_[class_idx]
        cov_det = self.covariance_det_[class_idx]
        prior = self.priors_[class_idx]
        
        # Quadratic term: -0.5 * (x - mu)^T * Sigma^-1 * (x - mu)
        X_centered = X - mean
        quad_term = -0.5 * np.sum((X_centered @ cov_inv) * X_centered, axis=1)
        
        # Log determinant term
        log_det_term = -0.5 * np.log(np.abs(cov_det) + 1e-300)
        
        # Prior term
        log_prior = np.log(prior + 1e-300)
        
        return quad_term + log_det_term + log_prior
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Compute discriminant for each class
        scores = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            scores[:, i] = self._compute_discriminant_function(X, i)
        
        # Softmax for probabilities
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


# ============================================================================
# PART 5: PYTORCH IMPLEMENTATION
# ============================================================================

def pytorch_implementation():
    """LDA and QDA implemented in PyTorch."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        print("\n" + "="*70)
        print("PYTORCH IMPLEMENTATION")
        print("="*70)
        
        class LDAPyTorch(nn.Module):
            """Linear Discriminant Analysis in PyTorch."""
            
            def __init__(self, n_features, n_classes, shrinkage=0.0):
                super().__init__()
                self.n_features = n_features
                self.n_classes = n_classes
                self.shrinkage = shrinkage
                
                # Parameters to be computed during fit
                self.register_buffer('means', torch.zeros(n_classes, n_features))
                self.register_buffer('priors', torch.zeros(n_classes))
                self.register_buffer('coef', torch.zeros(n_classes, n_features))
                self.register_buffer('intercept', torch.zeros(n_classes))
                
            def fit(self, X, y):
                """Fit the LDA model."""
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.long)
                
                n_samples = X.shape[0]
                classes = torch.unique(y)
                
                # Compute class statistics
                means = torch.zeros(self.n_classes, self.n_features)
                priors = torch.zeros(self.n_classes)
                
                for i, c in enumerate(classes):
                    mask = (y == c)
                    X_c = X[mask]
                    means[i] = X_c.mean(dim=0)
                    priors[i] = mask.sum().float() / n_samples
                
                # Compute pooled covariance
                Sw = torch.zeros(self.n_features, self.n_features)
                for i, c in enumerate(classes):
                    mask = (y == c)
                    X_c = X[mask]
                    X_centered = X_c - means[i]
                    Sw += X_centered.T @ X_centered
                    
                Sw = Sw / (n_samples - self.n_classes)
                
                # Apply shrinkage
                if self.shrinkage > 0:
                    trace = torch.trace(Sw) / self.n_features
                    Sw = (1 - self.shrinkage) * Sw + self.shrinkage * trace * torch.eye(self.n_features)
                
                # Compute coefficients
                Sw_inv = torch.linalg.inv(Sw)
                coef = means @ Sw_inv
                intercept = -0.5 * (means * (means @ Sw_inv)).sum(dim=1)
                intercept = intercept + torch.log(priors)
                
                # Store parameters
                self.means.copy_(means)
                self.priors.copy_(priors)
                self.coef.copy_(coef)
                self.intercept.copy_(intercept)
                
                return self
            
            def forward(self, X):
                """Compute discriminant scores."""
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32)
                return X @ self.coef.T + self.intercept
            
            def predict_proba(self, X):
                """Predict probabilities."""
                scores = self.forward(X)
                return F.softmax(scores, dim=1)
            
            def predict(self, X):
                """Predict class labels."""
                scores = self.forward(X)
                return torch.argmax(scores, dim=1)
        
        
        class QDAPyTorch(nn.Module):
            """Quadratic Discriminant Analysis in PyTorch."""
            
            def __init__(self, n_features, n_classes, reg_param=0.0):
                super().__init__()
                self.n_features = n_features
                self.n_classes = n_classes
                self.reg_param = reg_param
                
                # Parameters
                self.register_buffer('means', torch.zeros(n_classes, n_features))
                self.register_buffer('priors', torch.zeros(n_classes))
                # Store covariance inverses as list of tensors
                self.cov_invs = []
                self.cov_log_dets = []
                
            def fit(self, X, y):
                """Fit the QDA model."""
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.long)
                
                n_samples = X.shape[0]
                classes = torch.unique(y)
                
                means = torch.zeros(self.n_classes, self.n_features)
                priors = torch.zeros(self.n_classes)
                self.cov_invs = []
                self.cov_log_dets = []
                
                for i, c in enumerate(classes):
                    mask = (y == c)
                    X_c = X[mask]
                    n_c = X_c.shape[0]
                    
                    means[i] = X_c.mean(dim=0)
                    priors[i] = n_c / n_samples
                    
                    # Class covariance
                    X_centered = X_c - means[i]
                    cov = (X_centered.T @ X_centered) / (n_c - 1)
                    
                    # Regularization
                    if self.reg_param > 0:
                        trace = torch.trace(cov) / self.n_features
                        cov = (1 - self.reg_param) * cov + \
                              self.reg_param * trace * torch.eye(self.n_features)
                    
                    # Inverse and log determinant
                    cov_inv = torch.linalg.inv(cov)
                    log_det = torch.logdet(cov)
                    
                    self.cov_invs.append(cov_inv)
                    self.cov_log_dets.append(log_det)
                
                self.means.copy_(means)
                self.priors.copy_(priors)
                
                return self
            
            def forward(self, X):
                """Compute quadratic discriminant scores."""
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32)
                    
                n_samples = X.shape[0]
                scores = torch.zeros(n_samples, self.n_classes)
                
                for i in range(self.n_classes):
                    X_centered = X - self.means[i]
                    quad_term = -0.5 * (X_centered @ self.cov_invs[i] * X_centered).sum(dim=1)
                    log_det_term = -0.5 * self.cov_log_dets[i]
                    log_prior = torch.log(self.priors[i])
                    
                    scores[:, i] = quad_term + log_det_term + log_prior
                    
                return scores
            
            def predict_proba(self, X):
                """Predict probabilities."""
                scores = self.forward(X)
                return F.softmax(scores, dim=1)
            
            def predict(self, X):
                """Predict class labels."""
                scores = self.forward(X)
                return torch.argmax(scores, dim=1)
        
        # Test with Wine dataset
        wine = load_wine()
        X, y = wine.data, wine.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PyTorch LDA
        lda_torch = LDAPyTorch(X.shape[1], len(np.unique(y)), shrinkage=0.1)
        lda_torch.fit(X_train_scaled, y_train)
        y_pred_torch = lda_torch.predict(X_test_scaled).numpy()
        
        print("\nPyTorch LDA Results:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_torch):.4f}")
        
        # PyTorch QDA
        qda_torch = QDAPyTorch(X.shape[1], len(np.unique(y)), reg_param=0.1)
        qda_torch.fit(X_train_scaled, y_train)
        y_pred_qda_torch = qda_torch.predict(X_test_scaled).numpy()
        
        print("\nPyTorch QDA Results:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_qda_torch):.4f}")
        
        return lda_torch, qda_torch
        
    except ImportError:
        print("\nPyTorch not installed. Skipping PyTorch implementation.")
        return None, None


# ============================================================================
# PART 6: TENSORFLOW IMPLEMENTATION
# ============================================================================

def tensorflow_implementation():
    """LDA and QDA implemented in TensorFlow."""
    try:
        import tensorflow as tf
        
        print("\n" + "="*70)
        print("TENSORFLOW IMPLEMENTATION")
        print("="*70)
        
        class LDATensorFlow:
            """Linear Discriminant Analysis in TensorFlow."""
            
            def __init__(self, n_features, n_classes, shrinkage=0.0):
                self.n_features = n_features
                self.n_classes = n_classes
                self.shrinkage = shrinkage
                
                # Parameters
                self.means = None
                self.priors = None
                self.coef = None
                self.intercept = None
                
            def fit(self, X, y):
                """Fit the LDA model."""
                X = tf.constant(X, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.int32)
                
                n_samples = tf.shape(X)[0]
                
                means_list = []
                priors_list = []
                
                for c in range(self.n_classes):
                    mask = tf.equal(y, c)
                    X_c = tf.boolean_mask(X, mask)
                    means_list.append(tf.reduce_mean(X_c, axis=0))
                    priors_list.append(tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.float32) / 
                                      tf.cast(n_samples, tf.float32))
                
                self.means = tf.stack(means_list)
                self.priors = tf.stack(priors_list)
                
                # Pooled covariance
                Sw = tf.zeros((self.n_features, self.n_features))
                for c in range(self.n_classes):
                    mask = tf.equal(y, c)
                    X_c = tf.boolean_mask(X, mask)
                    X_centered = X_c - self.means[c]
                    Sw = Sw + tf.matmul(tf.transpose(X_centered), X_centered)
                
                n_samples_float = tf.cast(n_samples, tf.float32)
                Sw = Sw / (n_samples_float - self.n_classes)
                
                # Shrinkage
                if self.shrinkage > 0:
                    trace = tf.linalg.trace(Sw) / self.n_features
                    Sw = (1 - self.shrinkage) * Sw + \
                         self.shrinkage * trace * tf.eye(self.n_features)
                
                # Coefficients
                Sw_inv = tf.linalg.inv(Sw)
                self.coef = tf.matmul(self.means, Sw_inv)
                
                intercept = -0.5 * tf.reduce_sum(
                    self.means * tf.matmul(self.means, Sw_inv), axis=1
                )
                self.intercept = intercept + tf.math.log(self.priors)
                
                return self
            
            def predict_proba(self, X):
                """Predict probabilities."""
                X = tf.constant(X, dtype=tf.float32)
                scores = tf.matmul(X, tf.transpose(self.coef)) + self.intercept
                return tf.nn.softmax(scores)
            
            def predict(self, X):
                """Predict class labels."""
                proba = self.predict_proba(X)
                return tf.argmax(proba, axis=1)
        
        
        class QDATensorFlow:
            """Quadratic Discriminant Analysis in TensorFlow."""
            
            def __init__(self, n_features, n_classes, reg_param=0.0):
                self.n_features = n_features
                self.n_classes = n_classes
                self.reg_param = reg_param
                
                self.means = None
                self.priors = None
                self.cov_invs = None
                self.cov_log_dets = None
                
            def fit(self, X, y):
                """Fit the QDA model."""
                X = tf.constant(X, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.int32)
                
                n_samples = tf.shape(X)[0]
                n_samples_float = tf.cast(n_samples, tf.float32)
                
                means_list = []
                priors_list = []
                cov_invs_list = []
                cov_log_dets_list = []
                
                for c in range(self.n_classes):
                    mask = tf.equal(y, c)
                    X_c = tf.boolean_mask(X, mask)
                    n_c = tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.float32)
                    
                    mean = tf.reduce_mean(X_c, axis=0)
                    prior = n_c / n_samples_float
                    
                    # Class covariance
                    X_centered = X_c - mean
                    cov = tf.matmul(tf.transpose(X_centered), X_centered) / (n_c - 1)
                    
                    # Regularization
                    if self.reg_param > 0:
                        trace = tf.linalg.trace(cov) / self.n_features
                        cov = (1 - self.reg_param) * cov + \
                              self.reg_param * trace * tf.eye(self.n_features)
                    
                    cov_inv = tf.linalg.inv(cov)
                    log_det = tf.linalg.logdet(cov)
                    
                    means_list.append(mean)
                    priors_list.append(prior)
                    cov_invs_list.append(cov_inv)
                    cov_log_dets_list.append(log_det)
                
                self.means = tf.stack(means_list)
                self.priors = tf.stack(priors_list)
                self.cov_invs = tf.stack(cov_invs_list)
                self.cov_log_dets = tf.stack(cov_log_dets_list)
                
                return self
            
            def predict_proba(self, X):
                """Predict probabilities."""
                X = tf.constant(X, dtype=tf.float32)
                n_samples = tf.shape(X)[0]
                
                scores_list = []
                for c in range(self.n_classes):
                    X_centered = X - self.means[c]
                    quad_term = -0.5 * tf.reduce_sum(
                        tf.matmul(X_centered, self.cov_invs[c]) * X_centered, axis=1
                    )
                    log_det_term = -0.5 * self.cov_log_dets[c]
                    log_prior = tf.math.log(self.priors[c])
                    
                    scores_list.append(quad_term + log_det_term + log_prior)
                
                scores = tf.stack(scores_list, axis=1)
                return tf.nn.softmax(scores)
            
            def predict(self, X):
                """Predict class labels."""
                proba = self.predict_proba(X)
                return tf.argmax(proba, axis=1)
        
        # Test with Wine dataset
        wine = load_wine()
        X, y = wine.data, wine.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # TensorFlow LDA
        lda_tf = LDATensorFlow(X.shape[1], len(np.unique(y)), shrinkage=0.1)
        lda_tf.fit(X_train_scaled, y_train)
        y_pred_tf = lda_tf.predict(X_test_scaled).numpy()
        
        print("\nTensorFlow LDA Results:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_tf):.4f}")
        
        # TensorFlow QDA
        qda_tf = QDATensorFlow(X.shape[1], len(np.unique(y)), reg_param=0.1)
        qda_tf.fit(X_train_scaled, y_train)
        y_pred_qda_tf = qda_tf.predict(X_test_scaled).numpy()
        
        print("\nTensorFlow QDA Results:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred_qda_tf):.4f}")
        
        return lda_tf, qda_tf
        
    except ImportError:
        print("\nTensorFlow not installed. Skipping TensorFlow implementation.")
        return None, None


# ============================================================================
# PART 7: MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main demonstration comparing all implementations."""
    
    print("="*70)
    print("LDA & QDA: COMPLETE IMPLEMENTATION WITH ALL FEATURES")
    print("="*70)
    
    # Load Wine dataset (real-world data)
    print("\n" + "-"*70)
    print("LOADING WINE DATASET")
    print("-"*70)
    
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Feature names: {feature_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # =========================================================================
    # FEATURE STANDARDIZATION
    # =========================================================================
    print("\n" + "-"*70)
    print("FEATURE STANDARDIZATION")
    print("-"*70)
    
    # From scratch
    scaler_scratch = StandardScalerFromScratch()
    X_train_scaled = scaler_scratch.fit_transform(X_train)
    X_test_scaled = scaler_scratch.transform(X_test)
    
    # Sklearn comparison
    scaler_sklearn = StandardScaler()
    X_train_scaled_sk = scaler_sklearn.fit_transform(X_train)
    X_test_scaled_sk = scaler_sklearn.transform(X_test)
    
    print("\nStandardization comparison (first feature):")
    print(f"  Scratch - Mean: {scaler_scratch.mean_[0]:.4f}, Std: {scaler_scratch.std_[0]:.4f}")
    print(f"  Sklearn - Mean: {scaler_sklearn.mean_[0]:.4f}, Std: {scaler_sklearn.scale_[0]:.4f}")
    print(f"  Max difference in transformed data: {np.max(np.abs(X_train_scaled - X_train_scaled_sk)):.2e}")
    
    # =========================================================================
    # FEATURE SELECTION
    # =========================================================================
    print("\n" + "-"*70)
    print("FEATURE SELECTION")
    print("-"*70)
    
    # Fisher's criterion (from scratch)
    selector_fisher = FeatureSelectorFromScratch(method='fisher', k=8)
    X_train_selected = selector_fisher.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector_fisher.transform(X_test_scaled)
    
    print("\nFisher's Discriminant Scores (from scratch):")
    for i, score in enumerate(selector_fisher.scores_):
        print(f"  {feature_names[i]}: {score:.4f}")
    
    print(f"\nSelected features (top 8): {[feature_names[i] for i in selector_fisher.selected_features_]}")
    
    # F-statistic comparison with sklearn
    selector_sklearn = SelectKBest(f_classif, k=8)
    X_train_selected_sk = selector_sklearn.fit_transform(X_train_scaled, y_train)
    
    print("\nF-statistic comparison with sklearn:")
    selector_f = FeatureSelectorFromScratch(method='f_statistic', k=8)
    selector_f.fit(X_train_scaled, y_train)
    print(f"  Scratch selected: {sorted(selector_f.selected_features_)}")
    print(f"  Sklearn selected: {sorted(selector_sklearn.get_support(indices=True))}")
    
    # =========================================================================
    # LDA: SCRATCH VS SKLEARN
    # =========================================================================
    print("\n" + "-"*70)
    print("LINEAR DISCRIMINANT ANALYSIS (LDA)")
    print("-"*70)
    
    # LDA from scratch (with regularization)
    print("\n--- LDA From Scratch ---")
    lda_scratch = LDAFromScratch(n_components=2, shrinkage=0.1)
    lda_scratch.fit(X_train_scaled, y_train)
    y_pred_lda_scratch = lda_scratch.predict(X_test_scaled)
    
    print(f"Shrinkage parameter: 0.1")
    print(f"Number of components: {lda_scratch.n_components}")
    print(f"Class priors: {lda_scratch.priors_}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_lda_scratch):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lda_scratch, target_names=wine.target_names))
    
    # LDA with sklearn
    print("\n--- LDA Scikit-learn ---")
    lda_sklearn = LinearDiscriminantAnalysis(shrinkage=0.1, solver='lsqr')
    lda_sklearn.fit(X_train_scaled, y_train)
    y_pred_lda_sklearn = lda_sklearn.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lda_sklearn):.4f}")
    
    # Compare predictions
    agreement = np.mean(y_pred_lda_scratch == y_pred_lda_sklearn)
    print(f"\nPrediction agreement (scratch vs sklearn): {agreement:.2%}")
    
    # =========================================================================
    # QDA: SCRATCH VS SKLEARN
    # =========================================================================
    print("\n" + "-"*70)
    print("QUADRATIC DISCRIMINANT ANALYSIS (QDA)")
    print("-"*70)
    
    # QDA from scratch (with regularization)
    print("\n--- QDA From Scratch ---")
    qda_scratch = QDAFromScratch(reg_param=0.1)
    qda_scratch.fit(X_train_scaled, y_train)
    y_pred_qda_scratch = qda_scratch.predict(X_test_scaled)
    
    print(f"Regularization parameter: 0.1")
    print(f"Class priors: {qda_scratch.priors_}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_qda_scratch):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_qda_scratch, target_names=wine.target_names))
    
    # QDA with sklearn
    print("\n--- QDA Scikit-learn ---")
    qda_sklearn = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda_sklearn.fit(X_train_scaled, y_train)
    y_pred_qda_sklearn = qda_sklearn.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_qda_sklearn):.4f}")
    
    # Compare predictions
    agreement_qda = np.mean(y_pred_qda_scratch == y_pred_qda_sklearn)
    print(f"\nPrediction agreement (scratch vs sklearn): {agreement_qda:.2%}")
    
    # =========================================================================
    # EFFECT OF REGULARIZATION
    # =========================================================================
    print("\n" + "-"*70)
    print("EFFECT OF REGULARIZATION")
    print("-"*70)
    
    shrinkage_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("\nLDA Accuracy vs Shrinkage:")
    print("-" * 40)
    for s in shrinkage_values:
        lda_temp = LDAFromScratch(shrinkage=s)
        lda_temp.fit(X_train_scaled, y_train)
        acc = lda_temp.score(X_test_scaled, y_test)
        print(f"  Shrinkage = {s:.1f}: {acc:.4f}")
    
    print("\nQDA Accuracy vs Regularization:")
    print("-" * 40)
    for r in shrinkage_values:
        qda_temp = QDAFromScratch(reg_param=r)
        qda_temp.fit(X_train_scaled, y_train)
        acc = qda_temp.score(X_test_scaled, y_test)
        print(f"  Reg param = {r:.1f}: {acc:.4f}")
    
    # =========================================================================
    # EFFECT OF FEATURE SELECTION
    # =========================================================================
    print("\n" + "-"*70)
    print("EFFECT OF FEATURE SELECTION ON PERFORMANCE")
    print("-"*70)
    
    k_values = [3, 5, 8, 10, 13]
    
    print("\nLDA Accuracy vs Number of Features:")
    print("-" * 40)
    for k in k_values:
        selector = FeatureSelectorFromScratch(method='fisher', k=k)
        X_train_k = selector.fit_transform(X_train_scaled, y_train)
        X_test_k = selector.transform(X_test_scaled)
        
        lda_temp = LDAFromScratch(shrinkage=0.1)
        lda_temp.fit(X_train_k, y_train)
        acc = lda_temp.score(X_test_k, y_test)
        print(f"  k = {k:2d} features: {acc:.4f}")
    
    print("\nQDA Accuracy vs Number of Features:")
    print("-" * 40)
    for k in k_values:
        selector = FeatureSelectorFromScratch(method='fisher', k=k)
        X_train_k = selector.fit_transform(X_train_scaled, y_train)
        X_test_k = selector.transform(X_test_scaled)
        
        qda_temp = QDAFromScratch(reg_param=0.1)
        qda_temp.fit(X_train_k, y_train)
        acc = qda_temp.score(X_test_k, y_test)
        print(f"  k = {k:2d} features: {acc:.4f}")
    
    # =========================================================================
    # DIMENSIONALITY REDUCTION WITH LDA
    # =========================================================================
    print("\n" + "-"*70)
    print("DIMENSIONALITY REDUCTION WITH LDA")
    print("-"*70)
    
    lda_reduction = LDAFromScratch(n_components=2, shrinkage=0.1)
    lda_reduction.fit(X_train_scaled, y_train)
    X_train_lda = lda_reduction.transform(X_train_scaled)
    X_test_lda = lda_reduction.transform(X_test_scaled)
    
    print(f"\nOriginal dimensions: {X_train_scaled.shape[1]}")
    print(f"Reduced dimensions: {X_train_lda.shape[1]}")
    
    # Compare with sklearn
    lda_sklearn_transform = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda_sk = lda_sklearn_transform.fit_transform(X_train_scaled, y_train)
    
    print(f"\nCorrelation between scratch and sklearn LDA components:")
    for i in range(min(2, X_train_lda.shape[1])):
        corr = np.corrcoef(X_train_lda[:, i], X_train_lda_sk[:, i])[0, 1]
        print(f"  Component {i+1}: {abs(corr):.4f}")
    
    # =========================================================================
    # PYTORCH & TENSORFLOW IMPLEMENTATIONS
    # =========================================================================
    pytorch_implementation()
    tensorflow_implementation()
    
    # =========================================================================
    # SUMMARY COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY COMPARISON")
    print("="*70)
    
    results = {
        'Method': ['LDA (Scratch)', 'LDA (Sklearn)', 'QDA (Scratch)', 'QDA (Sklearn)'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_lda_scratch),
            accuracy_score(y_test, y_pred_lda_sklearn),
            accuracy_score(y_test, y_pred_qda_scratch),
            accuracy_score(y_test, y_pred_qda_sklearn)
        ]
    }
    
    print("\n" + pd.DataFrame(results).to_string(index=False))
    
    print("\n" + "="*70)
    print("IMPLEMENTATION COMPLETE!")
    print("="*70)
    
    return {
        'lda_scratch': lda_scratch,
        'qda_scratch': qda_scratch,
        'scaler': scaler_scratch,
        'selector': selector_fisher
    }


if __name__ == "__main__":
    results = main()