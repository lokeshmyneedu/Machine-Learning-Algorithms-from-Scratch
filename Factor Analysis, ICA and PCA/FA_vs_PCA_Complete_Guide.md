# Factor Analysis vs PCA: Complete Guide with Rotation Methods

## Table of Contents
1. [Core Differences](#core-differences)
2. [Mathematical Models](#mathematical-models)
3. [When to Use Each](#when-to-use-each)
4. [Implementation Details](#implementation-details)
5. [Factor Rotation - Deep Dive](#factor-rotation)
6. [Practical Examples](#practical-examples)

---

## Core Differences

### Philosophical Difference

**PCA (Principal Component Analysis)**
- **Purpose**: Dimensionality reduction and data compression
- **Assumption**: All variance is meaningful
- **Mathematical nature**: Deterministic transformation
- **Model**: `X = Z * W^T` (no error term)

**Factor Analysis**
- **Purpose**: Discovering latent variables that cause observed correlations
- **Assumption**: Variance = Common variance (from factors) + Unique variance (error)
- **Mathematical nature**: Statistical model with distributional assumptions
- **Model**: `X = μ + L * F + ε` (explicit error term)

### Variance Decomposition

This is THE critical difference:

**PCA**:
```
Total Variance = Explained Variance (by components)
```
No distinction between types of variance.

**Factor Analysis**:
```
Var(x_i) = Communality (h²_i) + Uniqueness (ψ_i)
         = Σ(l²_ij)      + (error + specific variance)
```

---

## Mathematical Models

### PCA Model

```
X = Z * W^T

where:
- X: (n × p) observed data
- Z: (n × k) principal component scores
- W: (p × k) eigenvectors (loadings)
```

**Algorithm**:
1. Center data: `X_c = X - mean(X)`
2. Compute covariance: `Σ = (1/n) * X_c^T * X_c`
3. Eigendecomposition: `Σ = W * Λ * W^T`
4. Sort by eigenvalues (descending)
5. Keep top k eigenvectors

**PCA Loadings**:
```python
loadings = eigenvectors * sqrt(eigenvalues)
```
This makes them interpretable as correlations between variables and components.

### Factor Analysis Model

```
X = μ + L * F + ε

where:
- X: (n × p) observed variables
- μ: (p × 1) mean vector
- L: (p × k) factor loading matrix
- F: (n × k) latent factors (UNOBSERVED)
- ε: (n × p) unique factors/errors
```

**Key Assumptions**:
- `F ~ N(0, I)` - factors are standard normal
- `ε ~ N(0, Ψ)` - errors are independent with diagonal covariance Ψ
- `Cov(F, ε) = 0` - factors and errors are independent

**Covariance Structure**:
```
Σ = L * L^T + Ψ

where:
- Σ: sample covariance matrix
- L * L^T: common variance (from factors)
- Ψ: unique variance (diagonal matrix)
```

**Maximum Likelihood Estimation**:

The ML estimates maximize:
```
log L(L, Ψ | X) ∝ -n/2 * log|Σ| - n/2 * tr(S * Σ^-1)

where S is the sample covariance matrix
```

We use EM algorithm:
- **E-step**: Estimate factor scores given current L, Ψ
- **M-step**: Update L, Ψ given factor scores
- Iterate until convergence

---

## When to Use Each

### Use PCA When:

1. **Goal is data compression**
   - Reducing features for computational efficiency
   - Preprocessing for other algorithms
   - Visualization of high-dimensional data

2. **All variance is informative**
   - Clean, reliable measurements
   - Measurement error is minimal
   - Want to preserve maximum information

3. **No theoretical model**
   - Exploratory analysis
   - No hypotheses about latent structure
   - Just need orthogonal features

**Examples**:
- Image compression (pixels → components)
- Gene expression analysis (thousands of genes → principal components)
- Preprocessing for regression/classification

### Use Factor Analysis When:

1. **Goal is discovering latent constructs**
   - Believe unobserved factors cause observed variables
   - Want interpretable factors with real-world meaning
   - Testing theoretical models

2. **Measurement error is present**
   - Survey data, questionnaires, psychological tests
   - Multiple indicators measuring same construct
   - Want to separate signal from noise

3. **Theory-driven analysis**
   - Have hypotheses about underlying factors
   - Want to understand causality (latent → observed)
   - Need to test if data supports theory

**Examples**:
- Personality questionnaires (Big Five factors)
- Customer satisfaction surveys (latent satisfaction dimensions)
- Student test scores (underlying abilities)
- Medical symptom assessments (underlying conditions)

---

## Implementation Details

### Key Differences in Practice

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| **Loadings** | Eigenvectors × √eigenvalues | Estimated via ML/EM |
| **Scores** | Exact: `Z = (X-μ) * W` | Estimated: requires inversion |
| **Rotation** | Not typically rotated | ALWAYS rotated for interpretability |
| **Uniqueness** | Not computed | Explicitly modeled (Ψ) |
| **Assumptions** | Minimal (just need data) | Assumes normality, specific error structure |
| **Communalities** | Always = 1 (all variance used) | < 1 (partial variance explained) |

### Communalities and Uniqueness

**Communality (h²_i)**: Proportion of variance in variable i explained by common factors
```
h²_i = Σ l²_ij  (sum of squared loadings)
     j
```

**Uniqueness (ψ_i)**: Proportion NOT explained by common factors
```
ψ_i = 1 - h²_i
    = measurement error + specific variance
```

**Example**:
```
Test: Algebra
Loadings: [0.90, 0.10]
Communality: 0.90² + 0.10² = 0.81 + 0.01 = 0.82
Uniqueness: 1 - 0.82 = 0.18

Interpretation: 82% of algebra score variance is explained by 
                common factors (math & verbal ability)
                18% is test-specific + measurement error
```

---

## Factor Rotation - Deep Dive

### Why Rotation is Necessary

**The Problem with Unrotated Factors**:

Initial factor extraction (ML, Principal Axis) optimizes for variance but produces factors that are hard to interpret:

```
BEFORE ROTATION:
                Factor1    Factor2
Algebra         -1.013      0.853    ← Both factors matter!
Geometry        -1.004      0.898    ← Both factors matter!
Statistics      -1.106      0.474    ← Both factors matter!
Reading         -1.035     -0.836    ← Both factors matter!
Vocabulary      -0.989     -0.873    ← Both factors matter!
Essay           -1.108     -0.549    ← Both factors matter!

What do these factors represent? UNCLEAR!
```

**After Rotation**:
```
AFTER VARIMAX ROTATION:
                Factor1    Factor2
Algebra         -0.115      1.319    ← Clearly Factor2!
Geometry        -0.077      1.345    ← Clearly Factor2!
Statistics      -0.448      1.117    ← Mostly Factor2
Reading         -1.323      0.139    ← Clearly Factor1!
Vocabulary      -1.317      0.080    ← Clearly Factor1!
Essay           -1.172      0.394    ← Mostly Factor1

Now clear: Factor1 = Verbal, Factor2 = Math
```

### Geometric Interpretation

Imagine factors as axes in space. Each variable is a point defined by its loadings.

**Unrotated**: Axes positioned to maximize variance (not interpretable)
**Rotated**: Axes rotated to align with variable clusters (interpretable)

```
         Before Rotation              After Rotation
    
    F2 ↑                           F2 ↑
       |  • Geometry                  | • Geometry  
       | • Algebra                    | • Algebra
       |/                             |/• Statistics
    ---|--------• Statistics   -------|-------------→ F1
       |       /                  /   |
       |    • Essay         • Essay   |
       |  • Reading       • Reading   |
       | • Vocabulary    • Vocabulary |

The relationships between variables haven't changed,
only our view (coordinate system) has rotated!
```

### Types of Rotation

#### 1. Orthogonal Rotations (Factors remain uncorrelated)

**VARIMAX** (Most Popular)
- **Objective**: Maximize variance of squared loadings within each factor
- **Effect**: Each factor has few large loadings, many near-zero loadings
- **Mathematical Criterion**:
  ```
  V = Σ [Σ l⁴_ij - (1/p)(Σ l²_ij)²]
      j  i              i
  ```
- **Use when**: Factors should be independent, want simplest interpretation

**QUARTIMAX**
- **Objective**: Simplify variables (each loads on one factor)
- **Effect**: Often creates a "general factor"
- **Mathematical Criterion**:
  ```
  Q = Σ Σ l⁴_ij
      i j
  ```
- **Use when**: Rarely (tends to produce less useful solutions)

#### 2. Oblique Rotations (Factors can correlate)

**PROMAX** (Most Popular Oblique)
- **Method**: Two-stage process
  1. First apply Varimax
  2. Then allow factors to correlate
- **Power parameter κ**: Usually 2-4 (higher = more oblique)
- **Outputs**:
  - **Pattern Matrix**: Unique contribution of each factor
  - **Structure Matrix**: Total correlation with factors
  - **Factor Correlation Matrix (Φ)**: Correlations between factors

- **Use when**: Factors likely correlate in reality

### Varimax Algorithm Details

```python
Algorithm: Varimax Rotation
---------------------------

1. Initialize:
   - Start with loadings L (p × k)
   - Rotation matrix T = I (k × k identity)

2. Kaiser Normalization:
   For each variable i:
     h_i = sqrt(Σ l²_ij)  # communality
     l*_ij = l_ij / h_i   # normalize
   
3. Iterate until convergence:
   a. Compute rotated loadings: L_rot = L* @ T
   
   b. For each factor pair (j, j'):
      Compute rotation angle φ that maximizes:
      V = Σ l⁴_ij - (1/p)(Σ l²_ij)²
      
   c. Update rotation matrix using SVD:
      U, S, V^T = SVD(L*^T @ (L_rot³ - L_rot @ diag(Σ L²_rot) / p))
      T = U @ V^T
   
   d. Check convergence: |T_new - T_old| < tolerance

4. Denormalize:
   l_ij = l*_ij @ T × h_i
```

**Key Properties**:
- Preserves orthogonality: `T^T @ T = I`
- Preserves communalities: `Σ l²_ij` unchanged for each variable
- Total variance explained unchanged
- Only changes interpretability, not statistical properties

### Promax Algorithm Details

```python
Algorithm: Promax Rotation
--------------------------

Stage 1: Orthogonal Rotation
   L_varimax = Varimax(L)

Stage 2: Oblique Transformation
   
   1. Create target matrix (exaggerate simple structure):
      target_ij = sign(l_varimax_ij) * |l_varimax_ij|^κ
      
      where κ is typically 4
      
      Effect: 
        Large loading 0.7 → 0.7⁴ = 0.24 (relatively large)
        Small loading 0.2 → 0.2⁴ = 0.0016 (very tiny)
   
   2. Find transformation T such that:
      L @ T ≈ target
      
      Solve: T = (L^T @ L)^-1 @ L^T @ target
   
   3. Pattern matrix (unique loadings):
      L_pattern = L @ T
   
   4. Compute factor correlations:
      Normalize L_pattern
      Φ = L_normalized^T @ L_normalized
   
   5. Structure matrix (total correlations):
      L_structure = L_pattern @ Φ
```

**Interpreting Promax Output**:

1. **Pattern Matrix** (use for interpretation):
   - Shows unique contribution of each factor
   - Like regression coefficients
   - "How much does Factor j predict Variable i, controlling for other factors?"

2. **Structure Matrix**:
   - Shows total correlation with factors
   - Like simple correlations
   - "How much does Variable i correlate with Factor j?"

3. **Factor Correlation Matrix (Φ)**:
   - Shows correlations between factors
   - If |r| > 0.3: factors are moderately correlated
   - If |r| < 0.2: might as well use Varimax

### Choosing Rotation Method

**Decision Tree**:

```
Are factors theoretically independent?
│
├─ YES → Use ORTHOGONAL rotation
│         │
│         └─ VARIMAX (default choice)
│
├─ NO → Use OBLIQUE rotation
│        │
│        └─ PROMAX (default choice)
│
└─ UNSURE → Try BOTH
             │
             ├─ Run Promax
             │
             ├─ Check Φ (factor correlations)
             │
             ├─ If |r| < 0.2 for all pairs → Use Varimax
             │
             └─ If |r| > 0.3 for any pair → Use Promax
```

**Practical Advice**:
1. Always start with Varimax (simpler)
2. Check if oblique rotation improves interpretability
3. If Promax shows weak correlations, stick with Varimax
4. Report both if results differ substantially

---

## Practical Examples

### Example 1: Student Test Scores

**Scenario**: 6 tests measuring mathematical and verbal abilities

**True Structure** (known from data generation):
```
              Math    Verbal
Algebra       0.90    0.10
Geometry      0.85    0.15
Statistics    0.70    0.40
Reading       0.10    0.90
Vocabulary    0.05    0.85
Essay         0.30    0.80
```

**Results**:

#### PCA:
```
PC1 explains 65.96% variance
PC2 explains 19.25% variance
Total: 85.21%

Loadings hard to interpret - mixed contributions
```

#### Factor Analysis (Unrotated):
```
All tests load on both factors
Difficult to identify what factors represent
```

#### Factor Analysis (Varimax):
```
              Factor1  Factor2  Communality
Algebra       -0.115   1.319    1.75
Geometry      -0.077   1.345    1.82
Statistics    -0.448   1.117    1.45
Reading       -1.323   0.139    1.77
Vocabulary    -1.317   0.080    1.74
Essay         -1.172   0.394    1.53

CLEAR STRUCTURE:
- Factor1 = Verbal Ability (Reading, Vocabulary, Essay)
- Factor2 = Math Ability (Algebra, Geometry, Statistics)

Communalities show 75-82% of variance explained by factors
```

#### Factor Analysis (Promax):
```
Factor correlation: r = 0.035 (negligible)
Since correlation is weak, Varimax is sufficient
Pattern matrix similar to Varimax
```

### Example 2: When to Use Each

#### Use PCA:
**Image Compression**
- Input: 64×64 pixel images (4096 dimensions)
- Goal: Compress to 100 dimensions
- Method: Keep top 100 PCs that capture 95% variance
- Why PCA?: All variance (including noise) might be important for reconstruction

#### Use FA:
**Customer Satisfaction Survey**
- Input: 20 questions about shopping experience
- Goal: Identify latent satisfaction dimensions
- Hypothesized factors:
  - Product Quality
  - Customer Service
  - Value for Money
  - Shopping Experience
- Why FA?: Want to discover underlying constructs, measurement error present

---

## Mathematical Derivations

### Communality Calculation

For variable i with loadings [l_i1, l_i2, ..., l_ik]:

```
h²_i = Σ l²_ij
       j=1 to k

Example:
Algebra: loadings = [0.90, 0.10]
h²_algebra = 0.90² + 0.10² = 0.81 + 0.01 = 0.82

Interpretation: 82% of algebra variance explained by factors
```

### Factor Score Estimation

Unlike PCA where scores are exact, FA scores must be estimated:

**Regression Method**:
```
F̂ = (L^T Ψ^-1 L + I)^-1 L^T Ψ^-1 (X - μ)
```

where:
- F̂: estimated factor scores (n × k)
- L: factor loadings (p × k)
- Ψ: uniqueness diagonal matrix (p × p)
- X: observed data (n × p)

This is the posterior mean under the Bayesian interpretation.

### Variance Explained by Factors

For rotated solution:

```
Variance explained by factor j:
V_j = Σ l²_ij  (sum down column j)
      i=1 to p

Proportion of total variance:
P_j = V_j / (Σ h²_i + Σ ψ_i)
           i        i
```

Note: After rotation, factors may explain different amounts of variance than before rotation, but total communality stays constant.

---

## Summary

### Key Takeaways

1. **PCA vs FA Distinction**:
   - PCA: Data reduction (all variance matters)
   - FA: Latent variable discovery (separate common from unique variance)

2. **When Rotation is Essential**:
   - Unrotated factors are hard to interpret
   - Rotation creates "simple structure"
   - Makes each variable load strongly on one factor

3. **Rotation Preserves**:
   - Communalities (variance explained per variable)
   - Total variance explained
   - Relationships between variables

4. **Rotation Changes**:
   - Interpretability (makes it better!)
   - Which variables load on which factors
   - Factor correlations (for oblique rotations)

5. **Practical Workflow**:
   1. Decide: PCA or FA?
   2. Extract factors (ML, Principal Axis)
   3. Choose rotation (Varimax first, then try Promax)
   4. Interpret loadings > 0.4 as "important"
   5. Name factors based on high-loading variables
   6. Check communalities and uniqueness
   7. Validate in new sample

### Final Recommendations

- **For exploration**: Try both PCA and FA, compare results
- **For latent constructs**: Use FA with rotation (Varimax or Promax)
- **For compression**: Use PCA
- **For theory testing**: Use FA with confirmatory approach
- **Always rotate**: Except for PCA or when factors are pre-defined
- **Report both**: Unrotated and rotated solutions for transparency

---

## References

- Thurstone, L.L. (1947). Multiple Factor Analysis
- Kaiser, H.F. (1958). The Varimax Criterion for Analytic Rotation in Factor Analysis
- Hendrickson & White (1964). Promax: A Quick Method for Rotation to Oblique Simple Structure
- Jollife, I.T. (2002). Principal Component Analysis
