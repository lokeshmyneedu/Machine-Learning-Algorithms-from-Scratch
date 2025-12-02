import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 1. Generate Dummy Data (Replace this with pd.read_csv('your_file.csv'))
np.random.seed(42)
n_samples = 200

# Feature 1: Gaussian (Normal) Distribution
# This represents "Height" or "Test Scores"
feature_gaussian = np.random.normal(loc=50, scale=10, size=n_samples)

# Feature 2: Skewed (Non-Gaussian) Distribution
# This represents "Income" (most are low, few are very high)
feature_skewed = np.random.exponential(scale=10, size=n_samples)

# Target: Linearly separable based on feature_gaussian
# If feature > 50, class is likely 1, else 0
target = (feature_gaussian + np.random.normal(0, 5, n_samples) > 50).astype(int)

df = pd.DataFrame({
    'Feature_Gaussian': feature_gaussian,
    'Feature_Skewed': feature_skewed,
    'Target': target
})

# ---------------------------------------------------------
# CHECK 1: Are the inputs Gaussian? (For GDA)
# ---------------------------------------------------------

def check_gaussian(data, col_name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # A. Histogram (Visual Check)
    # Gaussian = Bell Curve shape
    sns.histplot(data[col_name], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title(f'Histogram of {col_name}')
    
    # B. Q-Q Plot (Visual Check)
    # Gaussian = Dots fall on the red line
    stats.probplot(data[col_name], dist="norm", plot=ax[1])
    ax[1].set_title(f'Q-Q Plot of {col_name}')
    
    plt.tight_layout()
    plt.show()
    
    # C. Statistical Test (Shapiro-Wilk)
    # p-value > 0.05 means it IS Gaussian
    stat, p = stats.shapiro(data[col_name])
    print(f"--- Shapiro-Wilk Test for {col_name} ---")
    print(f"Statistics={stat:.3f}, p-value={p:.3f}")
    if p > 0.05:
        print("Result: Gaussian (Fail to reject H0)")
    else:
        print("Result: NOT Gaussian (Reject H0)")
    print("-" * 30)

print("### CHECKING FEATURE DISTRIBUTIONS ###\n")
check_gaussian(df, 'Feature_Gaussian')
check_gaussian(df, 'Feature_Skewed')

# ---------------------------------------------------------
# CHECK 2: Is the Boundary Linear? (For Logistic Regression)
# ---------------------------------------------------------

def check_linear_separability(df, x_col, y_col):
    plt.figure(figsize=(8, 5))
    
    # Scatter plot
    # Good for Logistic = You can draw a straight line between Blue and Orange
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=y_col, palette='coolwarm', s=100, alpha=0.6)
    plt.title(f'Scatter Plot: {x_col} vs Target')
    plt.yticks([0, 1])
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

print("\n### CHECKING LINEAR SEPARABILITY ###\n")
check_linear_separability(df, 'Feature_Gaussian', 'Target')