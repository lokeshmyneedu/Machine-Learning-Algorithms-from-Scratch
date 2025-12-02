import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. HELPER: MANUAL FILTER METHOD
# ==========================================
def filter_features_manual(X, y, feature_names, threshold=0.2):
    """
    Calculates correlation between features and target. 
    Drops features below the absolute threshold.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    corr = df.corr()['target'].abs()
    
    # Select features where correlation > threshold (ignore target itself)
    selected = corr[corr > threshold].index.tolist()
    if 'target' in selected: selected.remove('target')
    
    print(f"[Manual Filter] Dropped: {set(feature_names) - set(selected)}")
    
    # Return filtered array and the list of kept names
    return df[selected].values, selected

# ==========================================
# 2. CUSTOM LINEAR REGRESSION CLASS
# ==========================================
class CustomLinearRegression:
    def __init__(self, learning_rate: float = 0.001, n_iterations: int = 1000, 
                 regularization: Optional[str] = None, lambda_reg: float = 0.01):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.loss_history = []

    def _feature_scaling(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        X_scaled = self._feature_scaling(X)

        for _ in range(self.n_iterations):
            # Shuffle per epoch
            indices = np.random.permutation(n_samples)
            X_shuf, y_shuf = X_scaled[indices], y[indices]

            for i in range(0, n_samples, batch_size):
                X_b = X_shuf[i:i+batch_size]
                y_b = y_shuf[i:i+batch_size]
                m_batch = X_b.shape[0]

                # Prediction
                y_pred = np.dot(X_b, self.weights) + self.bias
                error = y_pred - y_b

                # Gradients
                dw = (1/m_batch) * np.dot(X_b.T, error)
                db = (1/m_batch) * np.sum(error)

                # Regularization / Embedded Selection
                if self.regularization == 'l2':
                    dw += self.lambda_reg * self.weights
                elif self.regularization == 'l1':
                    # L1 Gradient: pushes small weights towards zero
                    dw += self.lambda_reg * np.sign(self.weights)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = (X - self.mean) / (self.std + 1e-8)
        return np.dot(X_scaled, self.weights) + self.bias

# ==========================================
# 3. MAIN EXECUTION & COMPARISON
# ==========================================
if __name__ == "__main__":
    # Load Data
    data = fetch_california_housing()
    X_raw, y = data.data, data.target
    feat_names = data.feature_names
    
    # Global Split
    X_train_full, X_test_full, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
    results = []

    # -----------------------------------
    # A. CUSTOM: FILTER METHOD (Manual)
    # -----------------------------------
    # Step 1: Filter features based on correlation
    X_filtered_train, kept_names = filter_features_manual(X_train_full, y_train, feat_names, threshold=0.1)
    
    # We must filter the Test set using the SAME names we kept in Train
    df_test = pd.DataFrame(X_test_full, columns=feat_names)
    X_filtered_test = df_test[kept_names].values
    
    # Step 2: Train Model
    model_custom_filter = CustomLinearRegression(learning_rate=0.001, n_iterations=500)
    model_custom_filter.fit(X_filtered_train, y_train)
    preds = model_custom_filter.predict(X_filtered_test)
    
    results.append({
        "Method": "Custom Filter (Corr)",
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "Features_Used": len(kept_names)
    })

    # -----------------------------------
    # B. CUSTOM: EMBEDDED METHOD (L1)
    # -----------------------------------
    # Use ALL features, but high L1 regularization to kill weak ones
    model_custom_l1 = CustomLinearRegression(learning_rate=0.001, n_iterations=500, regularization='l1', lambda_reg=0.1)
    model_custom_l1.fit(X_train_full, y_train)
    preds = model_custom_l1.predict(X_test_full)
    
    # Count non-zero weights
    feats_l1 = np.sum(np.abs(model_custom_l1.weights) > 0.05) # Threshold for "zero"
    
    results.append({
        "Method": "Custom Embedded (L1)",
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "Features_Used": feats_l1
    })

    # -----------------------------------
    # C. SKLEARN: FILTER METHOD
    # -----------------------------------
    # Pipeline: Scale -> SelectKBest -> SGDRegressor
    pipe_filter = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_regression, k=4)), # Force pick top 4
        ('model', SGDRegressor(random_state=42, max_iter=500))
    ])
    
    pipe_filter.fit(X_train_full, y_train)
    preds = pipe_filter.predict(X_test_full)
    
    results.append({
        "Method": "Sklearn Filter (KBest)",
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "Features_Used": 4 # We explicitly asked for 4
    })

    # -----------------------------------
    # D. SKLEARN: EMBEDDED METHOD
    # -----------------------------------
    # Pipeline: Scale -> SelectFromModel(Lasso) -> SGDRegressor
    # We use Lasso to find features, then SGD to train on them
    pipe_embedded = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(Lasso(alpha=0.05, random_state=42))),
        ('model', SGDRegressor(random_state=42, max_iter=500))
    ])
    
    pipe_embedded.fit(X_train_full, y_train)
    preds = pipe_embedded.predict(X_test_full)
    
    # Check how many SelectFromModel kept
    selected_mask = pipe_embedded.named_steps['selector'].get_support()
    
    results.append({
        "Method": "Sklearn Embedded (Lasso)",
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "Features_Used": np.sum(selected_mask)
    })

    # ==========================================
    # 4. FINAL OUTPUT & VISUALIZATION
    # ==========================================
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("FINAL COMPARISON: CUSTOM VS SKLEARN FEATURE SELECTION")
    print("="*70)
    print(results_df.to_string(index=False))
    print("-" * 70)
    print(f"Original Feature Count: {X_train_full.shape[1]}")
    print("="*70)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bar Chart for MSE (Lower is better)
    bars = ax1.bar(results_df['Method'], results_df['MSE'], color=['#3498db', '#2980b9', '#e74c3c', '#c0392b'])
    ax1.set_title('Model Error (MSE) - Lower is Better')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_ylim(0, max(results_df['MSE']) * 1.2)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    # 2. Bar Chart for Feature Count (Efficiency)
    bars2 = ax2.bar(results_df['Method'], results_df['Features_Used'], color='gray')
    ax2.axhline(y=X_train_full.shape[1], color='black', linestyle='--', label='Original Count')
    ax2.set_title('Features Used (out of 8)')
    ax2.set_ylabel('Count')
    ax2.legend()
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()