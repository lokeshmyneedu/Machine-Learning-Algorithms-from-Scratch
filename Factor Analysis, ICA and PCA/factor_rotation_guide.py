"""
FACTOR ROTATION: COMPLETE MATHEMATICAL GUIDE
=============================================

This document provides a comprehensive explanation of factor rotation,
including mathematical derivations, algorithms, and practical insights.

Author: Lokesh's Learning Journey
Topic: Factor Analysis - Rotation Methods
"""

# ============================================================================
# PART 1: WHY ROTATION IS NEEDED
# ============================================================================

"""
THE FUNDAMENTAL PROBLEM WITH UNROTATED FACTORS
-----------------------------------------------

When you perform Factor Analysis (or even PCA), the initial solution is
mathematically correct but often difficult to interpret. Here's why:

INITIAL EXTRACTION (e.g., Maximum Likelihood, Principal Axis):
- Factors are extracted to maximize variance explained
- First factor captures maximum possible variance
- Second factor captures maximum remaining variance (orthogonal to first)
- And so on...

Result: Factors are OPTIMAL for variance but TERRIBLE for interpretation.

Example: Student Test Scores
Before Rotation:
                Factor1    Factor2
Algebra         -1.013      0.853    <- Both factors matter
Geometry        -1.004      0.898    <- Both factors matter
Statistics      -1.106      0.474    <- Both factors matter
Reading         -1.035     -0.836    <- Both factors matter
Vocabulary      -0.989     -0.873    <- Both factors matter
Essay           -1.108     -0.549    <- Both factors matter

What do Factor1 and Factor2 represent? Hard to say!

After Varimax Rotation:
                Factor1    Factor2
Algebra         -0.115      1.319    <- Clearly Factor2!
Geometry        -0.077      1.345    <- Clearly Factor2!
Statistics      -0.448      1.117    <- Mostly Factor2
Reading         -1.323      0.138    <- Clearly Factor1!
Vocabulary      -1.317      0.080    <- Clearly Factor1!
Essay           -1.172      0.394    <- Mostly Factor1

Now it's clear:
- Factor1 = Verbal Ability (loads on Reading, Vocabulary, Essay)
- Factor2 = Math Ability (loads on Algebra, Geometry, Statistics)

This is called "SIMPLE STRUCTURE"!
"""

# ============================================================================
# PART 2: WHAT IS ROTATION?
# ============================================================================

"""
ROTATION: GEOMETRIC INTERPRETATION
-----------------------------------

Think of factors as axes in a coordinate system. Each variable (test score)
is a point in this space, defined by its loadings on each factor.

UNROTATED SOLUTION:
    Factor2 ↑
            |    • Geometry
            |  • Algebra
            | /
            |/________• Statistics
    --------|----------------→ Factor1
            |          \\
            |           • Essay
            |        • Reading
            |      • Vocabulary

The axes (factors) are positioned by the extraction algorithm to maximize
variance, but the clustering of variables is hard to see.

AFTER ROTATION (Varimax):
         Factor2 ↑
                 | • Geometry
                 | • Algebra
                 |/• Statistics
    -------------|-------------→ Factor1
             /   |
    • Essay /    |
    • Reading    |
    • Vocabulary |

Now we've ROTATED the axes so they align better with the variable clusters!
The position of each point (variable) relative to OTHER points hasn't changed,
only the reference axes have rotated.

KEY INSIGHT:
Rotation doesn't change the fundamental relationships between variables.
It only makes the factor structure easier to interpret by repositioning
the axes.
"""

# ============================================================================
# PART 3: TYPES OF ROTATION
# ============================================================================

"""
TWO MAIN CATEGORIES OF ROTATION
--------------------------------

1. ORTHOGONAL ROTATIONS (Factors remain uncorrelated)
   - Varimax
   - Quartimax
   - Equamax

2. OBLIQUE ROTATIONS (Factors can be correlated)
   - Promax
   - Oblimin
   - Quartimin

ORTHOGONAL vs OBLIQUE: Which to choose?

Use ORTHOGONAL when:
- You believe underlying factors are truly independent
- You want simpler interpretation (uncorrelated factors)
- Your theory suggests factors shouldn't correlate

Use OBLIQUE when:
- Factors might be correlated in reality
- You want the most accurate model (even if more complex)
- Example: Intelligence sub-factors are usually correlated

In practice, OBLIQUE rotations often give better simple structure
because real-world factors are rarely perfectly uncorrelated.
"""

# ============================================================================
# PART 4: VARIMAX ROTATION - DETAILED MATHEMATICS
# ============================================================================

"""
VARIMAX: THE MOST POPULAR ROTATION METHOD
------------------------------------------

Objective: Maximize the variance of SQUARED loadings within each factor

Mathematical Criterion:
    V = (1/p) * Σ[j=1 to k] Σ[i=1 to p] (l²ij - (1/p)Σl²ij)²

Simplified (after Kaiser normalization):
    V = Σ[j=1 to k] [Σ[i=1 to p] l⁴ij - (1/p)(Σ[i=1 to p] l²ij)²]

Where:
- p = number of variables
- k = number of factors
- lij = loading of variable i on factor j

INTUITION:
We want each factor to have:
- A few variables with LARGE (squared) loadings
- Most variables with SMALL (near-zero) loadings

This creates "simple structure": each variable loads strongly on ONE factor.

WHY SQUARED LOADINGS?
Squaring amplifies differences:
- 0.9² = 0.81 (stays large)
- 0.3² = 0.09 (becomes very small)
- 0.1² = 0.01 (becomes tiny)

This makes the distinction between "important" and "unimportant" loadings clearer.

ALGORITHM (Iterative):
---------------------

1. Start with initial loadings L (p × k)

2. Kaiser Normalization:
   For each row i:
   - Compute communality: h²i = Σj l²ij
   - Normalize: l*ij = lij / √h²i
   
   This ensures all variables contribute equally regardless of communality.

3. For each pair of factors (j, j'):
   
   a. Define:
      A = Σi (l*²ij - l*²ij')
      B = Σi (l*²ij + l*²ij')
      C = Σi (l*ij * l*ij')
      D = Σi (l*ij * l*ij'(l*²ij - l*²ij'))
   
   b. Compute rotation angle:
      tan(4φ) = (D - 2AB) / (A² - B² - 4C²)
   
   c. Apply 2×2 rotation matrix:
      [l*ij(new)]   [cos(φ)  -sin(φ)] [l*ij]
      [l*ij'(new)] = [sin(φ)   cos(φ)] [l*ij']

4. Repeat step 3 for all factor pairs until convergence

5. Denormalize:
   lij = l*ij * √h²i

CONVERGENCE CRITERION:
Stop when change in loadings < tolerance (e.g., 1e-6)

PROPERTIES:
-----------
- Orthogonal rotation (factors remain uncorrelated)
- Rotation matrix T is orthonormal: T^T * T = I
- Communalities are preserved: Σj l²ij unchanged for each variable
- Total variance explained unchanged
"""

import numpy as np
from scipy import linalg

def varimax_detailed(loadings, max_iter=100, tol=1e-6, verbose=True):
    """
    Varimax rotation with detailed step-by-step output.
    
    Parameters:
    -----------
    loadings : ndarray (p, k)
        Initial factor loadings
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print detailed steps
        
    Returns:
    --------
    rotated_loadings : ndarray (p, k)
        Rotated loadings
    rotation_matrix : ndarray (k, k)
        Orthogonal rotation matrix
    """
    n_vars, n_factors = loadings.shape
    
    if verbose:
        print("="*80)
        print("VARIMAX ROTATION: STEP-BY-STEP")
        print("="*80)
        print(f"Variables: {n_vars}, Factors: {n_factors}")
    
    # Step 1: Kaiser normalization
    if verbose:
        print("\nStep 1: Kaiser Normalization")
        print("-" * 40)
    
    h = np.sqrt(np.sum(loadings**2, axis=1))  # Communalities
    loadings_normalized = loadings / h[:, np.newaxis]
    
    if verbose:
        print(f"Communalities (h²): {h**2}")
        print(f"\nNormalized loadings:\n{loadings_normalized}")
    
    # Step 2: Initialize rotation matrix
    T = np.eye(n_factors)
    
    if verbose:
        print(f"\nInitial rotation matrix (identity):\n{T}")
    
    # Step 3: Iterative rotation
    for iteration in range(max_iter):
        if verbose and iteration < 3:  # Show first 3 iterations
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}")
            print(f"{'='*80}")
        
        # Current rotated loadings
        L = loadings_normalized @ T
        
        # Compute gradient using SVD
        # This is the standard algorithm for Varimax
        u, s, vt = linalg.svd(
            loadings_normalized.T @ (L**3 - L @ np.diag(np.sum(L**2, axis=0)) / n_vars)
        )
        
        # New rotation matrix
        T_new = u @ vt
        
        if verbose and iteration < 3:
            print(f"Current varimax criterion: {varimax_criterion(L):.6f}")
            print(f"Rotation matrix:\n{T_new}")
        
        # Check convergence
        change = np.max(np.abs(T - T_new))
        if change < tol:
            if verbose:
                print(f"\nConverged after {iteration + 1} iterations!")
                print(f"Final change: {change:.2e}")
            break
            
        T = T_new
    
    # Step 4: Apply rotation and denormalize
    rotated_loadings = loadings_normalized @ T
    rotated_loadings = rotated_loadings * h[:, np.newaxis]
    
    if verbose:
        print(f"\nFinal rotated loadings:\n{rotated_loadings}")
        print(f"\nFinal varimax criterion: {varimax_criterion(rotated_loadings / h[:, np.newaxis]):.6f}")
    
    return rotated_loadings, T


def varimax_criterion(loadings):
    """
    Compute the varimax criterion value.
    
    V = Σ[j=1 to k] [Σ[i=1 to p] l⁴ij - (1/p)(Σ[i=1 to p] l²ij)²]
    """
    n_vars, n_factors = loadings.shape
    criterion = 0.0
    
    for j in range(n_factors):
        sum_fourth_power = np.sum(loadings[:, j]**4)
        sum_squared = np.sum(loadings[:, j]**2)
        criterion += sum_fourth_power - (sum_squared**2) / n_vars
    
    return criterion


# ============================================================================
# PART 5: QUARTIMAX ROTATION
# ============================================================================

"""
QUARTIMAX: SIMPLIFY VARIABLES (ROWS)
-------------------------------------

While Varimax simplifies factors (columns), Quartimax simplifies variables (rows).

Objective: Maximize sum of fourth powers of ALL loadings

Mathematical Criterion:
    Q = Σ[i=1 to p] Σ[j=1 to k] l⁴ij

INTUITION:
For each variable, we want it to load strongly on ONE factor and weakly on all others.

Example:
Good Quartimax structure:
    Variable 1: [0.95, 0.05, 0.02]  <- Loads on Factor 1 only
    Variable 2: [0.03, 0.91, 0.04]  <- Loads on Factor 2 only
    Variable 3: [0.02, 0.06, 0.93]  <- Loads on Factor 3 only

Bad structure:
    Variable 1: [0.60, 0.55, 0.50]  <- Loads on all factors!

DIFFERENCE FROM VARIMAX:
- Varimax: Distributes variance across factors (column-wise optimization)
- Quartimax: Concentrates variance in variables (row-wise optimization)

PRACTICAL NOTE:
Quartimax often produces a "general factor" - one factor that loads on
everything. This is usually NOT what we want, so Varimax is more popular.
"""

def quartimax_detailed(loadings, max_iter=100, tol=1e-6):
    """
    Quartimax rotation.
    
    Maximizes: Q = Σ_i Σ_j l⁴_ij
    """
    n_vars, n_factors = loadings.shape
    
    # Kaiser normalization
    h = np.sqrt(np.sum(loadings**2, axis=1))
    loadings_normalized = loadings / h[:, np.newaxis]
    
    T = np.eye(n_factors)
    
    for iteration in range(max_iter):
        L = loadings_normalized @ T
        
        # Gradient for Quartimax
        u, s, vt = linalg.svd(loadings_normalized.T @ (L**3))
        
        T_new = u @ vt
        
        if np.max(np.abs(T - T_new)) < tol:
            break
            
        T = T_new
    
    # Denormalize
    rotated_loadings = loadings_normalized @ T * h[:, np.newaxis]
    
    return rotated_loadings, T


# ============================================================================
# PART 6: PROMAX ROTATION (OBLIQUE)
# ============================================================================

"""
PROMAX: THE MOST POPULAR OBLIQUE ROTATION
------------------------------------------

Promax is a two-stage method:
1. First perform Varimax (orthogonal)
2. Then allow factors to correlate by relaxing orthogonality

WHY OBLIQUE?
In reality, factors are often correlated. For example:
- Mathematical ability and Verbal ability often correlate (both need reasoning)
- Depression and Anxiety often correlate
- Extroversion and Agreeableness can correlate

By allowing correlation, we often get even SIMPLER structure than Varimax.

ALGORITHM:
----------

Stage 1: Varimax Rotation
   Get L_varimax (orthogonal solution)

Stage 2: Promax Transformation
   
   a. Create target matrix by raising loadings to power κ (typically κ=4):
      
      target_ij = sign(l_varimax_ij) * |l_varimax_ij|^κ
      
      This EXAGGERATES the simple structure from Varimax:
      - Large loadings (0.7) become HUGE (0.7^4 = 0.24)
      - Small loadings (0.2) become TINY (0.2^4 = 0.0016)
   
   b. Find transformation T such that:
      L_promax * Φ ≈ target
      
      Where Φ is the factor correlation matrix (not necessarily identity!)
   
   c. Solve for T using least squares:
      T = (L_initial^T * L_initial)^-1 * L_initial^T * target
   
   d. Compute pattern matrix (loadings):
      L_promax = L_initial * T
   
   e. Compute factor correlations Φ:
      Normalize L_promax to get structure matrix
      Φ = correlation between factors

PROMAX OUTPUT:
--------------
Unlike orthogonal rotations, Promax gives THREE matrices:

1. PATTERN MATRIX (L): Unique contribution of each factor
   "How much does Factor j predict Variable i, controlling for other factors?"
   These are like regression coefficients.

2. STRUCTURE MATRIX (S): Total correlation with factors
   S = L * Φ
   "How much does Variable i correlate with Factor j?"
   These are like simple correlations.

3. FACTOR CORRELATION MATRIX (Φ): Correlations between factors
   Shows how much factors overlap

INTERPRETATION:
For interpretation, use PATTERN matrix (L)!
It shows unique contributions of each factor.
"""

def promax_detailed(loadings, kappa=4, max_iter=100, tol=1e-6, verbose=True):
    """
    Promax rotation with detailed output.
    
    Parameters:
    -----------
    loadings : ndarray (p, k)
        Initial loadings
    kappa : float
        Power for target matrix (typically 2-4)
    verbose : bool
        Print details
    """
    if verbose:
        print("="*80)
        print("PROMAX ROTATION: STEP-BY-STEP")
        print("="*80)
    
    n_vars, n_factors = loadings.shape
    
    # Stage 1: Varimax rotation
    if verbose:
        print("\nStage 1: Varimax Rotation (orthogonal)")
        print("-" * 40)
    
    L_varimax, T_varimax = varimax_detailed(loadings, verbose=False)
    
    if verbose:
        print(f"Varimax rotated loadings:\n{L_varimax}")
    
    # Stage 2: Promax transformation
    if verbose:
        print(f"\nStage 2: Promax Transformation (κ={kappa})")
        print("-" * 40)
    
    # Create target matrix
    signs = np.sign(L_varimax)
    target = signs * (np.abs(L_varimax) ** kappa)
    
    if verbose:
        print(f"\nTarget matrix (exaggerated structure):\n{target}")
    
    # Oblique transformation
    # Solve: loadings * T ≈ target
    # T = (loadings^T * loadings)^-1 * loadings^T * target
    T_promax = linalg.lstsq(loadings, target)[0]
    
    # Pattern matrix (unique loadings)
    L_pattern = loadings @ T_promax
    
    if verbose:
        print(f"\nPattern matrix (unique contributions):\n{L_pattern}")
    
    # Normalize to get structure matrix and factor correlations
    # Structure = Pattern * Φ, where Φ is factor correlation
    
    # Compute factor correlations
    # This is approximate - more sophisticated methods exist
    L_normalized = L_pattern / np.sqrt(np.sum(L_pattern**2, axis=0))
    phi = L_normalized.T @ L_normalized
    
    # Structure matrix
    L_structure = L_pattern @ phi
    
    if verbose:
        print(f"\nFactor correlation matrix (Φ):\n{phi}")
        print(f"\nStructure matrix:\n{L_structure}")
        
        # Interpret correlations
        print("\nFactor Correlations:")
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                corr = phi[i, j]
                if abs(corr) > 0.3:
                    print(f"  Factor {i+1} ↔ Factor {j+1}: r = {corr:.3f} (MODERATE)")
                elif abs(corr) > 0.1:
                    print(f"  Factor {i+1} ↔ Factor {j+1}: r = {corr:.3f} (weak)")
                else:
                    print(f"  Factor {i+1} ↔ Factor {j+1}: r = {corr:.3f} (negligible)")
    
    return L_pattern, L_structure, phi, T_promax


# ============================================================================
# PART 7: PRACTICAL GUIDANCE
# ============================================================================

"""
WHICH ROTATION METHOD TO USE?
------------------------------

Decision Tree:
1. Do you have strong theoretical reasons to believe factors are uncorrelated?
   ├─ YES → Use ORTHOGONAL rotation
   │         ├─ Want simplest interpretation? → VARIMAX
   │         └─ Want to simplify variables? → QUARTIMAX
   │
   └─ NO or UNSURE → Use OBLIQUE rotation
                      └─ PROMAX (most common choice)

In Practice:
- Try BOTH Varimax and Promax
- Compare the solutions
- If Promax shows weak factor correlations (|r| < 0.2), use Varimax
- If Promax shows strong correlations (|r| > 0.3), use Promax

INTERPRETING ROTATED SOLUTIONS:
--------------------------------

For ORTHOGONAL rotations (Varimax, Quartimax):
1. Look at the loadings matrix
2. Identify which variables load strongly (|loading| > 0.4) on each factor
3. Name each factor based on these variables

For OBLIQUE rotations (Promax):
1. Use the PATTERN matrix for interpretation
2. Identify strong loadings (|loading| > 0.4)
3. Check factor correlations in Φ
4. Name factors based on unique contributions

SIMPLE STRUCTURE CRITERIA (Thurstone, 1947):
---------------------------------------------
Good rotation should have:
1. Each row (variable) has at least one near-zero loading
2. Each column (factor) has some near-zero loadings
3. Each factor pair should have different patterns of loadings
4. Small number of variables load on multiple factors
5. Small number of factors load on the same variables

COMMUNALITIES AFTER ROTATION:
------------------------------
IMPORTANT: Communalities (h²) are INVARIANT under rotation!

Before rotation: h²i = Σj l²ij
After rotation:  h²i = Σj l'²ij  (same value!)

This makes sense: rotation changes our VIEW of the space,
but doesn't change how much variance is explained for each variable.

WARNINGS:
---------
1. Heywood Cases: Sometimes loadings > 1.0 or communalities > 1.0
   - Usually indicates improper solution
   - Try reducing number of factors
   - Check for outliers or data issues

2. Rotation is Indeterminate:
   - Many rotations can achieve similar simple structure
   - Choice between Varimax/Promax can be arbitrary
   - Always check if results make theoretical sense!

3. Don't Over-Interpret:
   - Factor analysis is exploratory
   - Replication in new samples is crucial
   - Theory should guide interpretation, not just statistical fit
"""


# ============================================================================
# DEMONSTRATION WITH REAL EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FACTOR ROTATION DEMONSTRATION")
    print("="*80)
    
    # Create example loading matrix
    # Two factors: Math and Verbal ability
    # Before rotation - mixed loadings
    unrotated_loadings = np.array([
        [-1.013,  0.853],  # Algebra
        [-1.004,  0.898],  # Geometry
        [-1.106,  0.474],  # Statistics
        [-1.035, -0.836],  # Reading
        [-0.989, -0.873],  # Vocabulary
        [-1.108, -0.549],  # Essay
    ])
    
    test_names = ['Algebra', 'Geometry', 'Statistics', 'Reading', 'Vocabulary', 'Essay']
    
    print("\nOriginal (Unrotated) Loadings:")
    print("-" * 60)
    print(f"{'Test':<15} {'Factor1':>12} {'Factor2':>12}")
    print("-" * 60)
    for i, name in enumerate(test_names):
        print(f"{name:<15} {unrotated_loadings[i, 0]:>12.3f} {unrotated_loadings[i, 1]:>12.3f}")
    
    print("\n" + "="*80)
    print("APPLYING VARIMAX ROTATION")
    print("="*80)
    varimax_loadings, T_varimax = varimax_detailed(unrotated_loadings, verbose=True)
    
    print("\n" + "="*80)
    print("APPLYING PROMAX ROTATION")
    print("="*80)
    pattern, structure, phi, T_promax = promax_detailed(unrotated_loadings, kappa=4, verbose=True)
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print("\nVarimax (Orthogonal) vs Promax (Oblique):")
    print("-" * 80)
    print(f"{'Test':<15} {'Varimax F1':>12} {'Varimax F2':>12} {'Promax F1':>12} {'Promax F2':>12}")
    print("-" * 80)
    for i, name in enumerate(test_names):
        print(f"{name:<15} {varimax_loadings[i, 0]:>12.3f} {varimax_loadings[i, 1]:>12.3f} "
              f"{pattern[i, 0]:>12.3f} {pattern[i, 1]:>12.3f}")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
1. ROTATION IMPROVES INTERPRETABILITY
   - Unrotated: All tests load on both factors (confusing!)
   - Rotated: Clear separation between Math and Verbal tests

2. VARIMAX CREATES SIMPLE STRUCTURE
   - Each test loads primarily on ONE factor
   - Easy to name factors based on which variables load high

3. PROMAX ALLOWS FACTOR CORRELATION
   - When factors are correlated in reality, Promax can be better
   - Check Φ matrix to see correlation strength
   - Use pattern matrix for interpretation

4. BOTH PRESERVE COMMUNALITIES
   - Total variance explained stays the same
   - Only the VIEW of the factor space changes

5. CHOICE DEPENDS ON YOUR DATA & THEORY
   - Try both methods
   - Compare interpretability
   - Consider theoretical expectations
    """)
