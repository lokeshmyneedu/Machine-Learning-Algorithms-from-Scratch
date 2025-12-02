import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load your data (Example)
# df = pd.read_csv('your_data.csv')

# --- Generate Dummy Multivariate Data for demonstration ---
np.random.seed(42)
mean = [10, 50, 100]
cov = [[5, 2, 1], [2, 10, 4], [1, 4, 20]] 
data = np.random.multivariate_normal(mean, cov, 300)
df = pd.DataFrame(data, columns=['Feature_A', 'Feature_B', 'Feature_C'])
# ----------------------------------------------------------

# METHOD 1: VISUALIZE ALL FEATURES AT ONCE (Grid)
def plot_all_histograms(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    
    # Create a layout with as many subplots as there are columns
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    
    if n_cols == 1: axes = [axes] # Handle single column case
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
        axes[i].set_title(f'{col} (Univariate)')
    
    plt.tight_layout()
    plt.show()

# METHOD 2: CHECK FULL DATASET MATH (Multivariate)
def check_multivariate_normality(df):
    print("\nChecking Multivariate Normality (Mahalanobis Distance)...")
    
    # Mathematical magic to condense all features into one distance metric
    x = df.values
    covariance_inv = np.linalg.inv(np.cov(x, rowvar=False))
    mean_diff = x - np.mean(x, axis=0)
    
    # Calculate Mahalanobis distance squared
    md_squared = np.diag(np.dot(np.dot(mean_diff, covariance_inv), mean_diff.T))
    
    # Compare against Chi-Square Distribution (The theoretical shape)
    df_degrees = df.shape[1] 
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(md_squared, dist="chi2", sparams=(df_degrees,), plot=ax)
    ax.set_title(f"Multivariate Normality Check\n(If dots are on line -> GDA is perfect)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Run the checks
plot_all_histograms(df)
check_multivariate_normality(df)