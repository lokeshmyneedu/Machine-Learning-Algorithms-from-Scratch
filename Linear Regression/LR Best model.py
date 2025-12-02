import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# 1. Setup Data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split & Scale (Critical step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper function to report GridSearch results
def report_best_model(grid_search, model_name):
    print(f"--- {model_name} Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # We negate the score because sklearn returns negative MSE
    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"Best CV RMSE: {best_rmse:.4f}")
    
    # Evaluate on the unseen Test Set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test Set MSE: {test_mse:.4f}")
    print(f"Test Set R2:  {test_r2:.4f}\n")


# --- 1. Ridge Regression with GridSearchCV ---
# We tune 'alpha' (strength) and 'solver' (algorithm)
ridge_params = {
    'alpha': [0.01, 0.1, 1, 10, 100], 
    'solver': ['svd', 'cholesky', 'lsqr', 'sag'] 
}

print("Searching for optimal Ridge parameters...")
ridge_grid = GridSearchCV(
    Ridge(random_state=42), 
    ridge_params, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1 # Use all CPU cores
)
ridge_grid.fit(X_train_scaled, y_train)
report_best_model(ridge_grid, "Ridge")


# --- 2. Lasso Regression with GridSearchCV ---
# We tune 'alpha'. Note: Lasso doesn't support the same solvers as Ridge.
# Lasso uses Coordinate Descent by default.
lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10]
}

print("Searching for optimal Lasso parameters...")
lasso_grid = GridSearchCV(
    Lasso(random_state=42), 
    lasso_params, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
lasso_grid.fit(X_train_scaled, y_train)
report_best_model(lasso_grid, "Lasso")


# --- 3. Elastic Net with GridSearchCV ---
# We tune 'alpha' and 'l1_ratio' (the mix between Ridge and Lasso)
# l1_ratio = 1 is Lasso, 0 is Ridge.
elastic_params = {
    'alpha': [0.01, 0.1, 1],
    'l1_ratio': [0.2, 0.5, 0.8]
}

print("Searching for optimal ElasticNet parameters...")
elastic_grid = GridSearchCV(
    ElasticNet(random_state=42), 
    elastic_params, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
elastic_grid.fit(X_train_scaled, y_train)
report_best_model(elastic_grid, "ElasticNet")