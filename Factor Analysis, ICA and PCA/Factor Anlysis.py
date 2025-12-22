from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

# 1. Load Real Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names).iloc[:, :10] # First 10 features

# 2. Standardize (Mandatory for FA)
X_scaled = StandardScaler().fit_transform(X)

# 3. Apply Factor Analysis
# We ask for 2 factors based on our "Size" vs "Shape" hypothesis
fa = FactorAnalysis(n_components=2, random_state=42)
fa.fit(X_scaled)

# 4. View the "Hidden" structure
loadings = pd.DataFrame(fa.components_.T, columns=['Factor 1', 'Factor 2'], index=X.columns)
print(loadings)



# 1. Initialize the analyzer
fa = FactorAnalyzer(rotation=None)
fa.fit(X_scaled)

# 2. Get Eigenvalues
ev, v = fa.get_eigenvalues()

# 3. Create the Scree Plot
plt.scatter(range(1, X_scaled.shape[1]+1), ev)
plt.plot(range(1, X_scaled.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--') # Kaiser Criterion line
plt.grid()
plt.show()