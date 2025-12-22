"""
Factor Analysis vs PCA: Complete Implementation from Scratch
============================================================

This implementation covers:
1. PCA from scratch with mathematical details
2. Factor Analysis from scratch (ML estimation)
3. Factor Rotation methods (Varimax, Quartimax, Promax)
4. Detailed comparison on student test scores dataset
"""

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# ============================================================================
# PART 1: PCA IMPLEMENTATION FROM SCRATCH
# ============================================================================

class PCA_FromScratch:
    """
    Principal Component Analysis implemented from first principles.
    
    Mathematical Model:
    X = Z * W^T
    where:
    - X: (n × p) data matrix
    - Z: (n × k) principal component scores
    - W: (p × k) loadings (eigenvectors)
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # Eigenvectors (p × k)
        self.eigenvalues_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """
        Fit PCA model.
        
        Steps:
        1. Center the data: X_centered = X - mean(X)
        2. Compute covariance matrix: Σ = (1/n) * X_centered^T * X_centered
        3. Eigen decomposition: Σ = W * Λ * W^T
        4. Sort eigenvectors by eigenvalues (descending)
        """
        n_samples, n_features = X.shape
        
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # Σ = (1/n) * X^T * X
        cov_matrix = (X_centered.T @ X_centered) / n_samples
        
        # Step 3: Eigen decomposition
        # We use eigh for symmetric matrices (more stable)
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        
        # Step 4: Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        if self.n_components is None:
            self.n_components = n_features
            
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Variance explained
        total_var = np.sum(eigenvalues)
        self.explained_variance_ = self.eigenvalues_
        self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        
        return self
    
    def transform(self, X):
        """
        Transform data to PC space.
        
        Z = (X - μ) * W
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_loadings(self):
        """
        Get loadings matrix.
        
        In PCA, loadings = eigenvectors * sqrt(eigenvalues)
        This makes them comparable to correlations
        
        Returns: (p × k) matrix where p = features, k = components
        """
        # components_ is (p × k), eigenvalues is (k,)
        # We want each column j to be eigenvector_j * sqrt(eigenvalue_j)
        loadings = self.components_ * np.sqrt(self.eigenvalues_)
        return loadings


# ============================================================================
# PART 2: FACTOR ANALYSIS IMPLEMENTATION FROM SCRATCH
# ============================================================================

class FactorAnalysis_FromScratch:
    """
    Factor Analysis implemented from first principles.
    
    Mathematical Model:
    X = μ + L * F + ε
    
    where:
    - X: (n × p) observed variables
    - μ: (p × 1) mean vector
    - L: (p × k) factor loading matrix
    - F: (n × k) latent factors (unobserved)
    - ε: (n × p) unique factors/errors
    
    Variance decomposition:
    Var(x_i) = h_i² + ψ_i
    where:
    - h_i² = communality (variance explained by common factors)
    - ψ_i = uniqueness (specific variance + measurement error)
    
    We'll use Maximum Likelihood (ML) estimation.
    """
    
    def __init__(self, n_factors=2, max_iter=100, tol=1e-4):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        
        self.mean_ = None
        self.loadings_ = None  # L: (p × k)
        self.uniquenesses_ = None  # ψ: (p,)
        self.communalities_ = None  # h²: (p,)
        
    def fit(self, X):
        """
        Fit Factor Analysis model using ML estimation.
        
        Algorithm:
        1. Initialize with PCA loadings
        2. EM algorithm to estimate L and ψ
        3. Iterate until convergence
        """
        n_samples, n_features = X.shape
        
        # Step 1: Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute sample covariance matrix
        S = (X_centered.T @ X_centered) / n_samples
        
        # Step 2: Initialize with PCA
        eigenvalues, eigenvectors = linalg.eigh(S)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Initial loadings from first k eigenvectors
        L = eigenvectors[:, :self.n_factors] * np.sqrt(eigenvalues[:self.n_factors])
        
        # Initial uniquenesses
        psi = np.maximum(np.diag(S - L @ L.T), 0.01)
        
        # Step 3: EM Algorithm
        for iteration in range(self.max_iter):
            L_old = L.copy()
            
            # E-step: Compute expected factor scores
            # E[F|X] = (L^T * Ψ^-1 * L + I)^-1 * L^T * Ψ^-1 * X^T
            Psi_inv = np.diag(1.0 / psi)
            M = L.T @ Psi_inv @ L + np.eye(self.n_factors)
            M_inv = linalg.inv(M)
            
            # Factor scores: (n × k)
            F_scores = (M_inv @ L.T @ Psi_inv @ X_centered.T).T
            
            # M-step: Update parameters
            # Update L: L_new = (X^T * F) * (F^T * F)^-1
            L = (X_centered.T @ F_scores) @ linalg.inv(F_scores.T @ F_scores)
            
            # Update ψ: diagonal of (S - L * L^T)
            residual = S - L @ L.T
            psi = np.maximum(np.diag(residual), 0.01)
            
            # Check convergence
            if np.max(np.abs(L - L_old)) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Store results
        self.loadings_ = L
        self.uniquenesses_ = psi
        
        # Compute communalities
        # h_i² = sum of squared loadings for variable i
        self.communalities_ = np.sum(L**2, axis=1)
        
        return self
    
    def transform(self, X):
        """
        Estimate factor scores for new data.
        
        F = (L^T * Ψ^-1 * L + I)^-1 * L^T * Ψ^-1 * (X - μ)^T
        """
        X_centered = X - self.mean_
        Psi_inv = np.diag(1.0 / self.uniquenesses_)
        M = self.loadings_.T @ Psi_inv @ self.loadings_ + np.eye(self.n_factors)
        M_inv = linalg.inv(M)
        
        F_scores = (M_inv @ self.loadings_.T @ Psi_inv @ X_centered.T).T
        return F_scores
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_variance_explained(self):
        """
        Calculate variance explained by each factor.
        
        Variance explained by factor j = sum of squared loadings in column j
        """
        factor_variances = np.sum(self.loadings_**2, axis=0)
        total_variance = np.sum(self.communalities_) + np.sum(self.uniquenesses_)
        proportion_var = factor_variances / total_variance
        return factor_variances, proportion_var


# ============================================================================
# PART 3: FACTOR ROTATION METHODS
# ============================================================================

class FactorRotation:
    """
    Factor rotation methods to improve interpretability.
    
    Types of rotation:
    1. Orthogonal rotations (factors remain uncorrelated):
       - Varimax: Maximizes variance of squared loadings
       - Quartimax: Simplifies variables (rows)
       
    2. Oblique rotations (factors can be correlated):
       - Promax: Allows factor correlation
    """
    
    @staticmethod
    def varimax(loadings, max_iter=100, tol=1e-6):
        """
        Varimax rotation: Maximize variance of squared loadings.
        
        Objective: Maximize V = Σ_j var(l_ij²)
        
        This makes each factor have a few large loadings and many near-zero loadings,
        improving interpretability.
        
        Algorithm: Iterative orthogonal rotation
        """
        n_vars, n_factors = loadings.shape
        
        # Normalize loadings (Kaiser normalization)
        h = np.sqrt(np.sum(loadings**2, axis=1))
        loadings_normalized = loadings / h[:, np.newaxis]
        
        # Initialize rotation matrix
        T = np.eye(n_factors)
        
        for iteration in range(max_iter):
            # Current rotated loadings
            L_rot = loadings_normalized @ T
            
            # Compute gradient of varimax criterion
            # This is a simplified version; full derivation is complex
            u, s, vt = linalg.svd(
                loadings_normalized.T @ (L_rot**3 - L_rot @ np.diag(np.sum(L_rot**2, axis=0)) / n_vars)
            )
            
            # New rotation matrix
            T_new = u @ vt
            
            # Check convergence
            if np.max(np.abs(T - T_new)) < tol:
                break
                
            T = T_new
        
        # Apply rotation
        rotated_loadings = loadings_normalized @ T
        
        # Denormalize
        rotated_loadings = rotated_loadings * h[:, np.newaxis]
        
        return rotated_loadings, T
    
    @staticmethod
    def quartimax(loadings, max_iter=100, tol=1e-6):
        """
        Quartimax rotation: Simplifies variables (rows).
        
        Objective: Maximize Q = Σ_i Σ_j l_ij⁴
        
        This tries to make each variable load highly on one factor only.
        """
        n_vars, n_factors = loadings.shape
        
        # Normalize
        h = np.sqrt(np.sum(loadings**2, axis=1))
        loadings_normalized = loadings / h[:, np.newaxis]
        
        T = np.eye(n_factors)
        
        for iteration in range(max_iter):
            L_rot = loadings_normalized @ T
            
            # Gradient for quartimax
            u, s, vt = linalg.svd(
                loadings_normalized.T @ (L_rot**3)
            )
            
            T_new = u @ vt
            
            if np.max(np.abs(T - T_new)) < tol:
                break
                
            T = T_new
        
        rotated_loadings = loadings_normalized @ T
        rotated_loadings = rotated_loadings * h[:, np.newaxis]
        
        return rotated_loadings, T
    
    @staticmethod
    def promax(loadings, kappa=4, max_iter=100, tol=1e-6):
        """
        Promax rotation: Oblique rotation allowing factor correlation.
        
        Steps:
        1. Start with Varimax rotation (orthogonal)
        2. Raise loadings to power kappa (default=4)
        3. Find oblique rotation that approximates this target
        
        Result: Factors can be correlated, often gives simpler structure
        """
        # Step 1: Varimax rotation
        L_varimax, _ = FactorRotation.varimax(loadings)
        
        # Step 2: Create target matrix by raising to power
        # Sign preservation
        signs = np.sign(L_varimax)
        target = signs * (np.abs(L_varimax) ** kappa)
        
        # Step 3: Oblique rotation
        # Solve: L_promax * Φ ≈ target
        # where Φ is the factor correlation matrix
        
        # This is a simplification; full Promax is more involved
        n_vars, n_factors = loadings.shape
        
        # Procrustes rotation
        L_promax = loadings.copy()
        for iteration in range(max_iter):
            # Update transformation
            # Minimize ||target - L_promax||
            Transform = linalg.lstsq(L_promax, target)[0]
            L_new = loadings @ Transform
            
            if np.max(np.abs(L_promax - L_new)) < tol:
                break
                
            L_promax = L_new
        
        # Compute factor correlation matrix
        # Normalize factors to unit variance
        factor_std = np.sqrt(np.sum(L_promax**2, axis=0))
        L_promax_normalized = L_promax / factor_std
        
        # Correlation between factors (this is approximate)
        phi = np.corrcoef(L_promax.T)
        
        return L_promax, Transform, phi


# ============================================================================
# PART 4: CREATE STUDENT TEST SCORES DATASET
# ============================================================================

def create_student_data(n_students=200):
    """
    Create synthetic student test score data with known factor structure.
    
    Underlying model:
    - Factor 1: Mathematical Ability
    - Factor 2: Verbal Ability
    
    Tests:
    1. Algebra (Math heavy)
    2. Geometry (Math heavy)
    3. Statistics (Math moderate, some verbal)
    4. Reading Comprehension (Verbal heavy)
    5. Vocabulary (Verbal heavy)
    6. Essay Writing (Verbal heavy, some reasoning)
    """
    
    # True latent factors (standardized)
    math_ability = np.random.randn(n_students)
    verbal_ability = np.random.randn(n_students)
    
    # Add some correlation between abilities (r ≈ 0.3)
    verbal_ability = verbal_ability + 0.3 * math_ability
    
    # Normalize
    math_ability = (math_ability - math_ability.mean()) / math_ability.std()
    verbal_ability = (verbal_ability - verbal_ability.mean()) / verbal_ability.std()
    
    # True loadings structure
    true_loadings = np.array([
        [0.90, 0.10],  # Algebra: 90% math, 10% verbal
        [0.85, 0.15],  # Geometry: 85% math, 15% verbal
        [0.70, 0.40],  # Statistics: 70% math, 40% verbal
        [0.10, 0.90],  # Reading: 10% math, 90% verbal
        [0.05, 0.85],  # Vocabulary: 5% math, 85% verbal
        [0.30, 0.80],  # Essay: 30% math (logic), 80% verbal
    ])
    
    # Generate observed scores
    # Score = mean + loading_math * math_ability + loading_verbal * verbal_ability + error
    
    test_names = ['Algebra', 'Geometry', 'Statistics', 'Reading', 'Vocabulary', 'Essay']
    scores = np.zeros((n_students, 6))
    
    for i in range(6):
        # Base score (mean)
        base = 75
        
        # Common variance (from factors)
        common = (true_loadings[i, 0] * math_ability * 10 + 
                 true_loadings[i, 1] * verbal_ability * 10)
        
        # Unique variance (measurement error + test-specific factors)
        uniqueness_std = np.sqrt(1 - np.sum(true_loadings[i]**2)) * 10
        unique = np.random.randn(n_students) * uniqueness_std
        
        scores[:, i] = base + common + unique
        
        # Clip to 0-100 range
        scores[:, i] = np.clip(scores[:, i], 0, 100)
    
    return scores, test_names, true_loadings, (math_ability, verbal_ability)


# ============================================================================
# PART 5: COMPARISON AND VISUALIZATION
# ============================================================================

def compare_pca_fa(X, feature_names):
    """
    Comprehensive comparison of PCA and Factor Analysis.
    """
    print("="*80)
    print("COMPARING PCA AND FACTOR ANALYSIS")
    print("="*80)
    
    # Standardize data
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # -------------------------------------------------------------------------
    # PCA Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("1. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*80)
    
    pca = PCA_FromScratch(n_components=2)
    pca_scores = pca.fit_transform(X_std)
    pca_loadings = pca.get_loadings()
    
    print(f"\nEigenvalues: {pca.eigenvalues_}")
    print(f"Variance explained: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    print("\n--- PCA Loadings ---")
    print("(Correlation between original variables and principal components)")
    print(f"\n{'Variable':<15} {'PC1':>8} {'PC2':>8}")
    print("-" * 35)
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {pca_loadings[i, 0]:>8.3f} {pca_loadings[i, 1]:>8.3f}")
    
    # -------------------------------------------------------------------------
    # Factor Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("2. FACTOR ANALYSIS (Before Rotation)")
    print("="*80)
    
    fa = FactorAnalysis_FromScratch(n_factors=2)
    fa_scores = fa.fit_transform(X_std)
    
    factor_vars, prop_vars = fa.get_variance_explained()
    print(f"\nFactor variances: {factor_vars}")
    print(f"Proportion of variance: {prop_vars}")
    
    print("\n--- FA Loadings (Unrotated) ---")
    print(f"\n{'Variable':<15} {'Factor1':>10} {'Factor2':>10} {'Communality':>12} {'Uniqueness':>12}")
    print("-" * 65)
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {fa.loadings_[i, 0]:>10.3f} {fa.loadings_[i, 1]:>10.3f} "
              f"{fa.communalities_[i]:>12.3f} {fa.uniquenesses_[i]:>12.3f}")
    
    print(f"\nTotal communality: {np.sum(fa.communalities_):.3f}")
    print(f"Total uniqueness: {np.sum(fa.uniquenesses_):.3f}")
    
    # -------------------------------------------------------------------------
    # Factor Rotation
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("3. FACTOR ANALYSIS (After VARIMAX Rotation)")
    print("="*80)
    
    rotated_loadings_varimax, T_varimax = FactorRotation.varimax(fa.loadings_)
    
    print("\n--- Varimax Rotated Loadings ---")
    print(f"\n{'Variable':<15} {'Factor1':>10} {'Factor2':>10} {'Communality':>12}")
    print("-" * 55)
    for i, name in enumerate(feature_names):
        communality = np.sum(rotated_loadings_varimax[i]**2)
        print(f"{name:<15} {rotated_loadings_varimax[i, 0]:>10.3f} "
              f"{rotated_loadings_varimax[i, 1]:>10.3f} {communality:>12.3f}")
    
    print("\n--- Rotation Matrix (Varimax) ---")
    print(T_varimax)
    
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("4. FACTOR ANALYSIS (After PROMAX Rotation)")
    print("="*80)
    
    rotated_loadings_promax, T_promax, phi = FactorRotation.promax(fa.loadings_)
    
    print("\n--- Promax Rotated Loadings ---")
    print(f"\n{'Variable':<15} {'Factor1':>10} {'Factor2':>10}")
    print("-" * 40)
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {rotated_loadings_promax[i, 0]:>10.3f} "
              f"{rotated_loadings_promax[i, 1]:>10.3f}")
    
    print("\n--- Factor Correlation Matrix (Phi) ---")
    print("(Correlations between factors - only meaningful for oblique rotations)")
    print(phi)
    
    return {
        'pca': pca,
        'pca_loadings': pca_loadings,
        'pca_scores': pca_scores,
        'fa': fa,
        'fa_loadings_unrotated': fa.loadings_,
        'fa_loadings_varimax': rotated_loadings_varimax,
        'fa_loadings_promax': rotated_loadings_promax,
        'fa_scores': fa_scores,
        'phi': phi
    }


def visualize_results(results, feature_names):
    """
    Create comprehensive visualizations comparing PCA and FA.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA Loadings
    ax = axes[0, 0]
    pca_loadings = results['pca_loadings']
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, pca_loadings[i, 0], pca_loadings[i, 1],
                head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        ax.text(pca_loadings[i, 0]*1.1, pca_loadings[i, 1]*1.1, name,
               fontsize=9, ha='center')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Loadings')
    ax.grid(True, alpha=0.3)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    ax.add_patch(circle)
    
    # 2. FA Loadings (Unrotated)
    ax = axes[0, 1]
    fa_loadings = results['fa_loadings_unrotated']
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, fa_loadings[i, 0], fa_loadings[i, 1],
                head_width=0.05, head_length=0.05, fc='red', ec='red')
        ax.text(fa_loadings[i, 0]*1.1, fa_loadings[i, 1]*1.1, name,
               fontsize=9, ha='center')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Factor 1')
    ax.set_ylabel('Factor 2')
    ax.set_title('FA Loadings (Unrotated)')
    ax.grid(True, alpha=0.3)
    
    # 3. FA Loadings (Varimax Rotated)
    ax = axes[0, 2]
    fa_loadings_rot = results['fa_loadings_varimax']
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, fa_loadings_rot[i, 0], fa_loadings_rot[i, 1],
                head_width=0.05, head_length=0.05, fc='green', ec='green')
        ax.text(fa_loadings_rot[i, 0]*1.1, fa_loadings_rot[i, 1]*1.1, name,
               fontsize=9, ha='center')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Factor 1 (Rotated)')
    ax.set_ylabel('Factor 2 (Rotated)')
    ax.set_title('FA Loadings (Varimax Rotated)')
    ax.grid(True, alpha=0.3)
    
    # 4. PCA Scores
    ax = axes[1, 0]
    ax.scatter(results['pca_scores'][:, 0], results['pca_scores'][:, 1],
              alpha=0.5, s=30)
    ax.set_xlabel('PC1 Score')
    ax.set_ylabel('PC2 Score')
    ax.set_title('PCA: Student Scores')
    ax.grid(True, alpha=0.3)
    
    # 5. FA Scores
    ax = axes[1, 1]
    ax.scatter(results['fa_scores'][:, 0], results['fa_scores'][:, 1],
              alpha=0.5, s=30, color='red')
    ax.set_xlabel('Factor 1 Score')
    ax.set_ylabel('Factor 2 Score')
    ax.set_title('FA: Student Factor Scores')
    ax.grid(True, alpha=0.3)
    
    # 6. Communalities vs Uniquenesses
    ax = axes[1, 2]
    fa = results['fa']
    x = np.arange(len(feature_names))
    width = 0.35
    ax.bar(x - width/2, fa.communalities_, width, label='Communality', color='green', alpha=0.7)
    ax.bar(x + width/2, fa.uniquenesses_, width, label='Uniqueness', color='orange', alpha=0.7)
    ax.set_xlabel('Test')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Decomposition (FA)')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Generating student test score data...")
    X, feature_names, true_loadings, (math_ability, verbal_ability) = create_student_data(n_students=200)
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    print("\n--- True Factor Loadings (Known) ---")
    print(f"\n{'Test':<15} {'Math':>8} {'Verbal':>8}")
    print("-" * 35)
    for i, name in enumerate(feature_names):
        print(f"{name:<15} {true_loadings[i, 0]:>8.3f} {true_loadings[i, 1]:>8.3f}")
    
    # Compare methods
    results = compare_pca_fa(X, feature_names)
    
    # Create visualizations
    fig = visualize_results(results, feature_names)
    output_path = 'fa_vs_pca_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    
    # -------------------------------------------------------------------------
    # KEY INSIGHTS
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("KEY INSIGHTS AND INTERPRETATIONS")
    print("="*80)
    
    print("\n1. LOADINGS INTERPRETATION:")
    print("-" * 40)
    print("PCA: Components are orthogonal linear combinations optimized for variance.")
    print("     Interpretation is often difficult.")
    print("\nFA (Unrotated): Similar to PCA, hard to interpret.")
    print("\nFA (Varimax): Clear simple structure!")
    print("     - Factor 1: Loads heavily on Math tests (Algebra, Geometry, Statistics)")
    print("     - Factor 2: Loads heavily on Verbal tests (Reading, Vocabulary, Essay)")
    
    print("\n2. COMMUNALITIES:")
    print("-" * 40)
    fa = results['fa']
    print("These tell us how much of each test's variance is explained by common factors:")
    for i, name in enumerate(feature_names):
        percent = fa.communalities_[i] * 100
        print(f"  {name:<15}: {percent:>5.1f}% (rest is test-specific + measurement error)")
    
    print("\n3. UNIQUENESSES:")
    print("-" * 40)
    print("These represent variance NOT explained by common factors:")
    print("  - Measurement error")
    print("  - Test-specific factors (e.g., spatial reasoning for Geometry)")
    
    print("\n4. WHY ROTATION MATTERS:")
    print("-" * 40)
    print("Before rotation: Mixed loadings, hard to interpret")
    print("After Varimax: Each variable loads strongly on ONE factor")
    print("This is the 'simple structure' that makes factors interpretable!")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
