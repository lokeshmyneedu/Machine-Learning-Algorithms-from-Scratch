import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA
import torch
import tensorflow as tf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. PREPROCESSING FROM SCRATCH
# ==========================================

class StandardScalerScratch:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Implemented from scratch using NumPy.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Prevent division by zero
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class FisherScoreSelector:
    """
    Selects top k features based on Fisher Score (Ratio of Between-Class Variance 
    to Within-Class Variance). This is highly relevant for LDA/QDA.
    """
    def __init__(self, k=5):
        self.k = k
        self.selected_indices = None
        self.scores = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        # Calculate overall mean
        mean_overall = np.mean(X, axis=0)
        
        # Calculate Between-Class Variance (S_B) and Within-Class Variance (S_W)
        S_B = np.zeros(n_features)
        S_W = np.zeros(n_features)
        
        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)
            
            # S_B: Weighted squared difference between class mean and overall mean
            S_B += n_c * (mean_c - mean_overall)**2
            
            # S_W: Sum of squared differences within the class
            S_W += np.sum((X_c - mean_c)**2, axis=0)
            
        # Fisher Score
        # Add small epsilon to S_W to avoid division by zero
        self.scores = S_B / (S_W + 1e-8)
        
        # Select top k indices
        self.selected_indices = np.argsort(self.scores)[::-1][:self.k]
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# ==========================================
# 2. NUMPY IMPLEMENTATION (FROM SCRATCH)
# ==========================================

class LDAScratch:
    def __init__(self, reg_param=0.0):
        """
        Linear Discriminant Analysis from Scratch.
        reg_param: float (0 to 1). Regularization strength (shrinkage).
        """
        self.reg_param = reg_param
        self.classes = None
        self.priors = None
        self.means = None
        self.shared_cov = None
        self.inv_cov = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.shared_cov = np.zeros((n_features, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = X_c.shape[0] / n_samples
            self.means[idx] = np.mean(X_c, axis=0)
            
            # Update shared covariance (weighted sum of individual covariances)
            # We use the scatter matrix sum, then divide by (N - K) later or (N) depending on estimator
            # Here we follow sklearn's convention roughly
            cov_c = np.cov(X_c, rowvar=False, bias=True) * X_c.shape[0] # Weighted
            self.shared_cov += cov_c

        # Finalize shared covariance
        self.shared_cov /= n_samples

        # Regularization (Shrinkage): (1-alpha)*Sigma + alpha*I
        if self.reg_param > 0:
            identity = np.eye(n_features)
            self.shared_cov = (1 - self.reg_param) * self.shared_cov + self.reg_param * identity

        self.inv_cov = np.linalg.inv(self.shared_cov)

    def predict(self, X):
        # Discriminant function: delta_k(x) = x.T * inv_cov * mu_k - 0.5 * mu_k.T * inv_cov * mu_k + log(pi_k)
        scores = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            mean = self.means[idx]
            
            # Linear term
            linear = X @ self.inv_cov @ mean.T
            
            # Quadratic term (constant for the class)
            quad = 0.5 * (mean @ self.inv_cov @ mean.T)
            
            score = linear - quad + prior
            scores.append(score)
            
        scores = np.array(scores).T # Shape: (n_samples, n_classes)
        return self.classes[np.argmax(scores, axis=1)]

class QDAScratch:
    def __init__(self, reg_param=0.0):
        """
        Quadratic Discriminant Analysis from Scratch.
        reg_param: Regularization adds to diagonal of covariance matrices.
        """
        self.reg_param = reg_param
        self.classes = None
        self.priors = None
        self.means = None
        self.covs = [] # List to store covariance per class
        self.inv_covs = []
        self.log_dets = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = X_c.shape[0] / n_samples
            self.means[idx] = np.mean(X_c, axis=0)
            
            # Class specific covariance
            cov_c = np.cov(X_c, rowvar=False, bias=True)
            
            # Regularization
            if self.reg_param > 0:
                cov_c += self.reg_param * np.eye(n_features)
                
            self.covs.append(cov_c)
            self.inv_covs.append(np.linalg.inv(cov_c))
            self.log_dets.append(np.linalg.slogdet(cov_c)[1])

    def predict(self, X):
        scores = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            mean = self.means[idx]
            inv_cov = self.inv_covs[idx]
            log_det = self.log_dets[idx]
            
            # QDA Score: -0.5*log|Sigma_k| - 0.5*(x-mu_k).T * Sigma_k^-1 * (x-mu_k) + log(pi_k)
            
            # Efficient computation of Mahalanobis distance
            diff = X - mean # (N, features)
            # (N, F) @ (F, F) -> (N, F) dot (N, F) -> (N,)
            mahalanobis = np.sum((diff @ inv_cov) * diff, axis=1)
            
            score = -0.5 * log_det - 0.5 * mahalanobis + prior
            scores.append(score)
            
        scores = np.array(scores).T
        return self.classes[np.argmax(scores, axis=1)]


# ==========================================
# 3. PYTORCH IMPLEMENTATION
# ==========================================

class LDA_PyTorch:
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param
        
    def fit(self, X, y):
        # Expecting PyTorch tensors
        self.classes = torch.unique(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        self.means = []
        self.priors = []
        
        # Compute shared covariance
        self.shared_cov = torch.zeros(n_features, n_features, dtype=torch.float32)
        
        for c in self.classes:
            mask = (y == c)
            X_c = X[mask]
            
            prior = X_c.shape[0] / n_samples
            mean = torch.mean(X_c, dim=0)
            
            self.priors.append(prior)
            self.means.append(mean)
            
            # Center data
            X_centered = X_c - mean
            cov = (X_centered.T @ X_centered) / X_c.shape[0]
            # Weighted addition
            self.shared_cov += cov * prior # weighting by prior equivalent to total sum / N

        self.means = torch.stack(self.means)
        self.priors = torch.tensor(self.priors, dtype=torch.float32)
        
        # Regularize
        self.shared_cov += self.reg_param * torch.eye(n_features)
        self.inv_cov = torch.inverse(self.shared_cov)
        
    def predict(self, X):
        scores = []
        for i, c in enumerate(self.classes):
            mean = self.means[i]
            prior = torch.log(self.priors[i])
            
            # Linear Discriminant
            # x.T * Sigma^-1 * mu
            linear = X @ self.inv_cov @ mean 
            
            # Quadratic term
            quad = 0.5 * (mean @ self.inv_cov @ mean)
            
            score = linear - quad + prior
            scores.append(score)
            
        scores = torch.stack(scores, dim=1)
        return self.classes[torch.argmax(scores, dim=1)]

class QDA_PyTorch:
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param
        
    def fit(self, X, y):
        self.classes = torch.unique(y)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        self.means = []
        self.priors = []
        self.inv_covs = []
        self.log_dets = []
        
        for c in self.classes:
            mask = (y == c)
            X_c = X[mask]
            
            prior = X_c.shape[0] / n_samples
            mean = torch.mean(X_c, dim=0)
            
            self.priors.append(prior)
            self.means.append(mean)
            
            X_centered = X_c - mean
            cov = (X_centered.T @ X_centered) / (X_c.shape[0] - 1) # Unbiased
            
            # Regularize
            cov += self.reg_param * torch.eye(n_features)
            
            self.inv_covs.append(torch.inverse(cov))
            self.log_dets.append(torch.logdet(cov))
            
        self.priors = torch.tensor(self.priors)
        self.means = torch.stack(self.means)

    def predict(self, X):
        scores = []
        for i, c in enumerate(self.classes):
            mean = self.means[i]
            inv_cov = self.inv_covs[i]
            log_det = self.log_dets[i]
            prior = torch.log(self.priors[i])
            
            diff = X - mean
            # Batched Mahalanobis: diag(diff @ inv_cov @ diff.T)
            # More efficient: sum((diff @ inv_cov) * diff, dim=1)
            mahalanobis = torch.sum((diff @ inv_cov) * diff, dim=1)
            
            score = -0.5 * log_det - 0.5 * mahalanobis + prior
            scores.append(score)
            
        scores = torch.stack(scores, dim=1)
        return self.classes[torch.argmax(scores, dim=1)]


# ==========================================
# 4. TENSORFLOW IMPLEMENTATION
# ==========================================

class LDA_TensorFlow:
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param

    def fit(self, X, y):
        # Expecting tf.Tensors
        self.classes, _ = tf.unique(y)
        # Ensure sorted classes for consistency
        self.classes = tf.sort(self.classes)
        
        n_features = tf.cast(tf.shape(X)[1], tf.float32)
        n_samples = tf.cast(tf.shape(X)[0], tf.float32)
        
        means = []
        priors = []
        
        shared_cov = tf.zeros((tf.shape(X)[1], tf.shape(X)[1]), dtype=tf.float32)
        
        for c in self.classes:
            mask = tf.equal(y, c)
            X_c = tf.boolean_mask(X, mask)
            n_c = tf.cast(tf.shape(X_c)[0], tf.float32)
            
            prior = n_c / n_samples
            mean = tf.reduce_mean(X_c, axis=0)
            
            means.append(mean)
            priors.append(prior)
            
            X_centered = X_c - mean
            cov = tf.matmul(tf.transpose(X_centered), X_centered) / n_c
            shared_cov += cov * prior

        self.means = tf.stack(means)
        self.priors = tf.stack(priors)
        
        # Regularize
        eye = tf.eye(tf.shape(X)[1], dtype=tf.float32)
        shared_cov += self.reg_param * eye
        self.inv_cov = tf.linalg.inv(shared_cov)
        
    def predict(self, X):
        scores = []
        for i in range(len(self.classes)):
            mean = self.means[i]
            prior = tf.math.log(self.priors[i])
            
            linear = tf.linalg.matvec(X @ self.inv_cov, mean)
            quad = 0.5 * tf.tensordot(tf.linalg.matvec(self.inv_cov, mean), mean, axes=1)
            
            score = linear - quad + prior
            scores.append(score)
            
        scores = tf.stack(scores, axis=1)
        predictions = tf.argmax(scores, axis=1)
        return tf.gather(self.classes, predictions)

class QDA_TensorFlow:
    def __init__(self, reg_param=1e-4):
        self.reg_param = reg_param

    def fit(self, X, y):
        self.classes, _ = tf.unique(y)
        self.classes = tf.sort(self.classes)
        n_samples = tf.cast(tf.shape(X)[0], tf.float32)
        n_features = tf.shape(X)[1]
        
        self.means = []
        self.priors = []
        self.inv_covs = []
        self.log_dets = []
        
        eye = tf.eye(n_features, dtype=tf.float32)

        for c in self.classes:
            mask = tf.equal(y, c)
            X_c = tf.boolean_mask(X, mask)
            n_c = tf.cast(tf.shape(X_c)[0], tf.float32)
            
            prior = n_c / n_samples
            mean = tf.reduce_mean(X_c, axis=0)
            
            self.means.append(mean)
            self.priors.append(prior)
            
            X_centered = X_c - mean
            cov = tf.matmul(tf.transpose(X_centered), X_centered) / (n_c - 1)
            cov += self.reg_param * eye
            
            self.inv_covs.append(tf.linalg.inv(cov))
            # Log det for numerical stability
            self.log_dets.append(tf.linalg.logdet(cov))
            
    def predict(self, X):
        scores = []
        for i in range(len(self.classes)):
            mean = self.means[i]
            inv_cov = self.inv_covs[i]
            log_det = self.log_dets[i]
            prior = tf.math.log(self.priors[i])
            
            diff = X - mean
            # (N, F) * (F, F) -> (N, F)
            left = tf.matmul(diff, inv_cov)
            # Row-wise dot product
            mahalanobis = tf.reduce_sum(left * diff, axis=1)
            
            score = -0.5 * log_det - 0.5 * mahalanobis + prior
            scores.append(score)
            
        scores = tf.stack(scores, axis=1)
        predictions = tf.argmax(scores, axis=1)
        return tf.gather(self.classes, predictions)


# ==========================================
# 5. MAIN EXECUTION & COMPARISON
# ==========================================

def run_pipeline():
    print("--- 1. LOADING DATA (WINE DATASET) ---")
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"Original shape: {X.shape}")
    print(f"Classes: {np.unique(y)}\n")

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("--- 2. STANDARDIZATION & FEATURE SELECTION (SCRATCH) ---")
    # A. Standardization
    scaler = StandardScalerScratch()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # B. Feature Selection (Fisher Score)
    selector = FisherScoreSelector(k=5) # Select top 5 features
    X_train_sel = selector.fit_transform(X_train_std, y_train)
    X_test_sel = selector.transform(X_test_std)
    
    selected_feats = [feature_names[i] for i in selector.selected_indices]
    print(f"Top 5 Features selected by Fisher Score: {selected_feats}")
    print(f"Transformed shape: {X_train_sel.shape}\n")
    
    # ==========================
    # NUMPY SCRATCH RESULTS
    # ==========================
    print("--- 3. RESULTS: NUMPY (FROM SCRATCH) ---")
    # LDA
    lda_scratch = LDAScratch(reg_param=0.1)
    lda_scratch.fit(X_train_sel, y_train)
    lda_pred = lda_scratch.predict(X_test_sel)
    print(f"LDA Scratch Accuracy: {accuracy_score(y_test, lda_pred):.4f}")
    
    # QDA
    qda_scratch = QDAScratch(reg_param=0.1)
    qda_scratch.fit(X_train_sel, y_train)
    qda_pred = qda_scratch.predict(X_test_sel)
    print(f"QDA Scratch Accuracy: {accuracy_score(y_test, qda_pred):.4f}\n")

    # ==========================
    # SCIKIT-LEARN COMPARISON
    # ==========================
    print("--- 4. RESULTS: SCIKIT-LEARN COMPARISON ---")
    # Note: sklearn solver='lsqr' supports shrinkage for LDA
    sk_lda = SklearnLDA(solver='lsqr', shrinkage=0.1)
    sk_lda.fit(X_train_sel, y_train)
    sk_pred = sk_lda.predict(X_test_sel)
    print(f"Sklearn LDA Accuracy: {accuracy_score(y_test, sk_pred):.4f}")
    
    sk_qda = SklearnQDA(reg_param=0.1)
    sk_qda.fit(X_train_sel, y_train)
    sk_qda_pred = sk_qda.predict(X_test_sel)
    print(f"Sklearn QDA Accuracy: {accuracy_score(y_test, sk_qda_pred):.4f}\n")

    # ==========================
    # PYTORCH COMPARISON
    # ==========================
    print("--- 5. RESULTS: PYTORCH IMPLEMENTATION ---")
    # Convert numpy to torch tensors
    Xt_train = torch.tensor(X_train_sel, dtype=torch.float32)
    Xt_test = torch.tensor(X_test_sel, dtype=torch.float32)
    yt_train = torch.tensor(y_train, dtype=torch.long)
    
    lda_torch = LDA_PyTorch(reg_param=0.1)
    lda_torch.fit(Xt_train, yt_train)
    pt_lda_pred = lda_torch.predict(Xt_test).numpy()
    print(f"PyTorch LDA Accuracy: {accuracy_score(y_test, pt_lda_pred):.4f}")
    
    qda_torch = QDA_PyTorch(reg_param=0.1)
    qda_torch.fit(Xt_train, yt_train)
    pt_qda_pred = qda_torch.predict(Xt_test).numpy()
    print(f"PyTorch QDA Accuracy: {accuracy_score(y_test, pt_qda_pred):.4f}\n")

    # ==========================
    # TENSORFLOW COMPARISON
    # ==========================
    print("--- 6. RESULTS: TENSORFLOW IMPLEMENTATION ---")
    # Convert numpy to tf tensors
    Xtf_train = tf.convert_to_tensor(X_train_sel, dtype=tf.float32)
    Xtf_test = tf.convert_to_tensor(X_test_sel, dtype=tf.float32)
    ytf_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    
    lda_tf = LDA_TensorFlow(reg_param=0.1)
    lda_tf.fit(Xtf_train, ytf_train)
    tf_lda_pred = lda_tf.predict(Xtf_test).numpy()
    print(f"TensorFlow LDA Accuracy: {accuracy_score(y_test, tf_lda_pred):.4f}")
    
    qda_tf = QDA_TensorFlow(reg_param=0.1)
    qda_tf.fit(Xtf_train, ytf_train)
    tf_qda_pred = qda_tf.predict(Xtf_test).numpy()
    print(f"TensorFlow QDA Accuracy: {accuracy_score(y_test, tf_qda_pred):.4f}")

if __name__ == "__main__":
    run_pipeline()