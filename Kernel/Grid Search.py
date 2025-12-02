"""
GRID SEARCH CROSS-VALIDATION EXPLAINED
=======================================
A comprehensive guide to understanding hyperparameter tuning

This script provides:
1. Visual explanation of what cross-validation does
2. Step-by-step grid search implementation
3. Comparison of different CV strategies
4. Practical examples with real data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, KFold, StratifiedKFold,
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: WHAT IS CROSS-VALIDATION?
# ============================================================================

def explain_cross_validation():
    """
    Visual explanation of K-Fold Cross-Validation
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        WHAT IS CROSS-VALIDATION?                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE PROBLEM:                                                                ║
║  ───────────                                                                 ║
║  If we use a single train/test split to evaluate hyperparameters:           ║
║  • We might get lucky/unlucky with the split                                ║
║  • Results are not reliable or reproducible                                 ║
║  • We might overfit to the test set                                         ║
║                                                                              ║
║  THE SOLUTION: K-FOLD CROSS-VALIDATION                                      ║
║  ─────────────────────────────────────                                       ║
║  1. Split data into K equal parts (folds)                                   ║
║  2. Train K models, each time using K-1 folds for training                  ║
║  3. Test each model on the remaining fold                                   ║
║  4. Average the K scores for final evaluation                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Visual representation of 5-Fold CV
    fig, axes = plt.subplots(5, 1, figsize=(12, 8))
    fig.suptitle('5-Fold Cross-Validation Visualization', fontsize=14, fontweight='bold')
    
    colors_train = '#3498db'  # Blue for training
    colors_test = '#e74c3c'   # Red for testing
    
    for fold in range(5):
        ax = axes[fold]
        
        # Create 5 segments
        for i in range(5):
            if i == fold:
                color = colors_test
                label = 'Test' if fold == 0 and i == 0 else ''
            else:
                color = colors_train
                label = 'Train' if fold == 0 and i == 1 else ''
            
            ax.barh(0, 1, left=i, color=color, edgecolor='white', 
                   linewidth=2, height=0.6, label=label)
            ax.text(i + 0.5, 0, f'Fold {i+1}', ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
        
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(f'Iteration {fold+1}', fontsize=10)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if fold == 0:
            ax.legend(loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig('cv_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cv_visualization.png")


# ============================================================================
# PART 2: WHAT IS GRID SEARCH?
# ============================================================================

def explain_grid_search():
    """
    Explain what grid search does
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          WHAT IS GRID SEARCH?                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  THE CONCEPT:                                                                ║
║  ────────────                                                                ║
║  Grid Search exhaustively tries ALL combinations of hyperparameters         ║
║  from a predefined "grid" of values.                                        ║
║                                                                              ║
║  EXAMPLE for SVM with RBF kernel:                                           ║
║  ─────────────────────────────────                                           ║
║  Parameter Grid:                                                             ║
║    • C     = [0.1, 1, 10, 100]     (4 values)                               ║
║    • gamma = [0.001, 0.01, 0.1, 1] (4 values)                               ║
║                                                                              ║
║  Total combinations = 4 × 4 = 16 models to train                            ║
║                                                                              ║
║  With 5-fold CV: 16 × 5 = 80 model fits!                                    ║
║                                                                              ║
║  THE GRID LOOKS LIKE:                                                        ║
║  ────────────────────                                                        ║
║                                                                              ║
║         gamma=0.001  gamma=0.01  gamma=0.1  gamma=1                         ║
║        ┌───────────┬───────────┬──────────┬─────────┐                       ║
║  C=0.1 │  Model 1  │  Model 2  │ Model 3  │ Model 4 │                       ║
║        ├───────────┼───────────┼──────────┼─────────┤                       ║
║  C=1   │  Model 5  │  Model 6  │ Model 7  │ Model 8 │                       ║
║        ├───────────┼───────────┼──────────┼─────────┤                       ║
║  C=10  │  Model 9  │  Model 10 │ Model 11 │ Model 12│                       ║
║        ├───────────┼───────────┼──────────┼─────────┤                       ║
║  C=100 │  Model 13 │  Model 14 │ Model 15 │ Model 16│                       ║
║        └───────────┴───────────┴──────────┴─────────┘                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================================
# PART 3: STEP-BY-STEP GRID SEARCH FROM SCRATCH
# ============================================================================

def grid_search_from_scratch_detailed(X, y):
    """
    Implement grid search with cross-validation step by step
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              GRID SEARCH FROM SCRATCH - STEP BY STEP                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    }
    
    n_folds = 5
    
    print(f"Parameter Grid:")
    print(f"  C values: {param_grid['C']}")
    print(f"  gamma values: {param_grid['gamma']}")
    print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['gamma'])}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Total model fits: {len(param_grid['C']) * len(param_grid['gamma']) * n_folds}")
    print("\n" + "=" * 70)
    
    # Create stratified k-fold splitter
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store all results
    all_results = []
    best_score = -np.inf
    best_params = None
    
    combination_num = 0
    
    # Iterate through all parameter combinations
    for C in param_grid['C']:
        for gamma in param_grid['gamma']:
            combination_num += 1
            print(f"\n{'─' * 70}")
            print(f"COMBINATION {combination_num}: C={C}, gamma={gamma}")
            print("─" * 70)
            
            fold_scores = []
            
            # Perform K-Fold Cross-Validation
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                # Split data
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                # Train model with current parameters
                model = SVC(kernel='rbf', C=C, gamma=gamma)
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluate on validation fold
                score = model.score(X_val_fold, y_val_fold)
                fold_scores.append(score)
                
                print(f"  Fold {fold_idx + 1}: "
                      f"Train size={len(train_idx)}, "
                      f"Val size={len(val_idx)}, "
                      f"Accuracy={score:.4f}")
            
            # Calculate mean and std of fold scores
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            print(f"\n  ► Mean CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            # Store results
            all_results.append({
                'C': C,
                'gamma': gamma,
                'mean_score': mean_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            })
            
            # Track best parameters
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'C': C, 'gamma': gamma}
                print(f"  ★ NEW BEST SCORE!")
    
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE!")
    print("=" * 70)
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV Score: {best_score:.4f}")
    
    return all_results, best_params


# ============================================================================
# PART 4: VISUALIZE GRID SEARCH RESULTS
# ============================================================================

def visualize_grid_search_results(X, y):
    """
    Create comprehensive visualizations of grid search results
    """
    print("\nCreating Grid Search Visualizations...")
    
    # Perform grid search with scikit-learn
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }
    
    svm = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy',
                               return_train_score=True, n_jobs=-1)
    grid_search.fit(X, y)
    
    # Extract results
    results = grid_search.cv_results_
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Heatmap of CV scores
    ax1 = fig.add_subplot(2, 2, 1)
    scores_matrix = np.zeros((len(param_grid['C']), len(param_grid['gamma'])))
    
    for i, C in enumerate(param_grid['C']):
        for j, gamma in enumerate(param_grid['gamma']):
            # Find the index for this combination
            mask = (results['param_C'] == C) & (results['param_gamma'] == gamma)
            idx = np.where(mask)[0][0]
            scores_matrix[i, j] = results['mean_test_score'][idx]
    
    im = ax1.imshow(scores_matrix, cmap='YlGnBu', aspect='auto')
    ax1.set_xticks(range(len(param_grid['gamma'])))
    ax1.set_yticks(range(len(param_grid['C'])))
    ax1.set_xticklabels(param_grid['gamma'])
    ax1.set_yticklabels(param_grid['C'])
    ax1.set_xlabel('gamma', fontsize=12)
    ax1.set_ylabel('C', fontsize=12)
    ax1.set_title('CV Accuracy Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(param_grid['C'])):
        for j in range(len(param_grid['gamma'])):
            color = 'white' if scores_matrix[i, j] < 0.95 else 'black'
            ax1.text(j, i, f'{scores_matrix[i, j]:.3f}', ha='center', va='center',
                    color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax1, label='Accuracy')
    
    # Mark best parameters
    best_idx = np.unravel_index(scores_matrix.argmax(), scores_matrix.shape)
    ax1.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=3))
    
    # 2. Train vs Validation scores
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Sort by mean test score
    sorted_idx = np.argsort(results['mean_test_score'])[::-1][:15]  # Top 15
    
    x_pos = np.arange(len(sorted_idx))
    width = 0.35
    
    train_scores = results['mean_train_score'][sorted_idx]
    test_scores = results['mean_test_score'][sorted_idx]
    test_std = results['std_test_score'][sorted_idx]
    
    bars1 = ax2.bar(x_pos - width/2, train_scores, width, label='Train', color='#3498db')
    bars2 = ax2.bar(x_pos + width/2, test_scores, width, label='Validation', 
                   color='#e74c3c', yerr=test_std, capsize=3)
    
    ax2.set_xlabel('Parameter Combination (ranked)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Train vs Validation Scores (Top 15)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0.9, 1.02)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{i+1}' for i in range(len(sorted_idx))], fontsize=8)
    
    # 3. Effect of C (fixing gamma)
    ax3 = fig.add_subplot(2, 2, 3)
    
    for gamma in [0.01, 0.1, 1]:
        scores = []
        for C in param_grid['C']:
            mask = (results['param_C'] == C) & (results['param_gamma'] == gamma)
            idx = np.where(mask)[0][0]
            scores.append(results['mean_test_score'][idx])
        ax3.semilogx(param_grid['C'], scores, 'o-', label=f'gamma={gamma}', linewidth=2)
    
    ax3.set_xlabel('C (log scale)', fontsize=12)
    ax3.set_ylabel('CV Accuracy', fontsize=12)
    ax3.set_title('Effect of C Parameter', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Effect of gamma (fixing C)
    ax4 = fig.add_subplot(2, 2, 4)
    
    for C in [0.1, 1, 10]:
        scores = []
        for gamma in param_grid['gamma']:
            mask = (results['param_C'] == C) & (results['param_gamma'] == gamma)
            idx = np.where(mask)[0][0]
            scores.append(results['mean_test_score'][idx])
        ax4.semilogx(param_grid['gamma'], scores, 's-', label=f'C={C}', linewidth=2)
    
    ax4.set_xlabel('gamma (log scale)', fontsize=12)
    ax4.set_ylabel('CV Accuracy', fontsize=12)
    ax4.set_title('Effect of gamma Parameter', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grid_search_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: grid_search_analysis.png")
    
    return grid_search


# ============================================================================
# PART 5: UNDERSTANDING HYPERPARAMETERS
# ============================================================================

def explain_hyperparameters():
    """
    Explain what each hyperparameter does
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    UNDERSTANDING SVM HYPERPARAMETERS                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  C (REGULARIZATION PARAMETER)                                                ║
║  ────────────────────────────                                                ║
║  • Controls the trade-off between margin width and classification errors    ║
║  • LOW C (e.g., 0.1):  Wider margin, allows more misclassifications         ║
║                        → More regularization, simpler model                  ║
║                        → May underfit                                        ║
║  • HIGH C (e.g., 100): Narrower margin, penalizes misclassifications more   ║
║                        → Less regularization, complex model                  ║
║                        → May overfit                                         ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  LOW C                              HIGH C                          │    ║
║  │  ═══════                            ══════                          │    ║
║  │                                                                     │    ║
║  │    ○  ○                               ○  ○                          │    ║
║  │  ○      ○                           ○      ○                        │    ║
║  │     ───────────────              ╔═══════════════╗                  │    ║
║  │  ○   (wide margin)  ●           ○║ (tight margin)║  ●              │    ║
║  │     ───────────────              ╚═══════════════╝                  │    ║
║  │  ○      ○    ●                      ○      ○    ●                   │    ║
║  │    ○  ○   ●   ●                       ○  ○   ●   ●                  │    ║
║  │                                                                     │    ║
║  │  Allows some points                 Tries to classify              │    ║
║  │  on wrong side                      all points correctly            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  GAMMA (RBF KERNEL PARAMETER)                                                ║
║  ────────────────────────────                                                ║
║  • Defines how far the influence of a single training example reaches       ║
║  • LOW gamma:  Far reach, smooth decision boundary                          ║
║                → Points far away still influence the boundary               ║
║                → May underfit                                                ║
║  • HIGH gamma: Close reach, complex/wiggly decision boundary                ║
║                → Only nearby points influence the boundary                  ║
║                → May overfit                                                 ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  LOW gamma                          HIGH gamma                       │    ║
║  │  ══════════                         ═══════════                      │    ║
║  │                                                                     │    ║
║  │    ○  ○        ●                      ○  ○        ●                 │    ║
║  │  ○      ○    ●                      ○  ╭───╮  ○    ●                │    ║
║  │     ────────────  ●                   ╭╯   ╰────╮   ●              │    ║
║  │  ○              ● ●                 ○ │         │ ● ●              │    ║
║  │     ────────────                      ╰─────────╯                   │    ║
║  │  ○      ○   ●                       ○      ○   ●                    │    ║
║  │    ○  ○   ●   ●                       ○  ○   ●   ●                  │    ║
║  │                                                                     │    ║
║  │  Smooth boundary                    Complex boundary                │    ║
║  │  (may miss details)                 (may overfit noise)             │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  DEGREE (POLYNOMIAL KERNEL)                                                  ║
║  ──────────────────────────                                                  ║
║  • Controls the degree of the polynomial: K(x,y) = (x·y + coef0)^degree     ║
║  • Higher degree = more complex decision boundary                           ║
║  • degree=1 is essentially linear, degree=2 is quadratic, etc.              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def visualize_hyperparameter_effects(X, y):
    """
    Visualize how different hyperparameters affect the decision boundary
    """
    from sklearn.decomposition import PCA
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Different C values (with fixed gamma)
    C_values = [0.01, 0.1, 10, 100]
    gamma_fixed = 0.1
    
    for idx, C in enumerate(C_values):
        ax = axes[0, idx]
        model = SVC(kernel='rbf', C=C, gamma=gamma_fixed)
        model.fit(X_2d, y)
        
        # Create mesh
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='red', s=20, alpha=0.6)
        ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='blue', s=20, alpha=0.6)
        ax.set_title(f'C={C}\n(gamma={gamma_fixed} fixed)', fontsize=11)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
    
    # Different gamma values (with fixed C)
    gamma_values = [0.001, 0.01, 0.1, 1]
    C_fixed = 10
    
    for idx, gamma in enumerate(gamma_values):
        ax = axes[1, idx]
        model = SVC(kernel='rbf', C=C_fixed, gamma=gamma)
        model.fit(X_2d, y)
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='red', s=20, alpha=0.6)
        ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='blue', s=20, alpha=0.6)
        ax.set_title(f'gamma={gamma}\n(C={C_fixed} fixed)', fontsize=11)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
    
    plt.suptitle('Effect of Hyperparameters on Decision Boundary', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hyperparameter_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hyperparameter_effects.png")


# ============================================================================
# PART 6: VALIDATION CURVES - DEEP DIVE
# ============================================================================

def create_validation_curves(X, y):
    """
    Create validation curves showing train/test scores across parameter ranges
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Validation curve for C
    C_range = np.logspace(-3, 3, 20)
    train_scores, test_scores = validation_curve(
        SVC(kernel='rbf', gamma=0.1), X, y,
        param_name='C', param_range=C_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    ax = axes[0]
    ax.semilogx(C_range, train_scores.mean(axis=1), 'o-', color='#3498db',
                label='Training score', linewidth=2)
    ax.fill_between(C_range, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.2, color='#3498db')
    
    ax.semilogx(C_range, test_scores.mean(axis=1), 's-', color='#e74c3c',
                label='Cross-validation score', linewidth=2)
    ax.fill_between(C_range, test_scores.mean(axis=1) - test_scores.std(axis=1),
                    test_scores.mean(axis=1) + test_scores.std(axis=1),
                    alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('C (regularization)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Curve for C\n(gamma=0.1 fixed)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.01)
    
    # Add annotations
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='Optimal region')
    ax.annotate('Underfitting\n(High bias)', xy=(0.01, 0.94), fontsize=10,
                ha='center', color='gray')
    ax.annotate('Overfitting\n(High variance)', xy=(500, 0.94), fontsize=10,
                ha='center', color='gray')
    
    # Validation curve for gamma
    gamma_range = np.logspace(-4, 1, 20)
    train_scores, test_scores = validation_curve(
        SVC(kernel='rbf', C=10), X, y,
        param_name='gamma', param_range=gamma_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    ax = axes[1]
    ax.semilogx(gamma_range, train_scores.mean(axis=1), 'o-', color='#3498db',
                label='Training score', linewidth=2)
    ax.fill_between(gamma_range, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.2, color='#3498db')
    
    ax.semilogx(gamma_range, test_scores.mean(axis=1), 's-', color='#e74c3c',
                label='Cross-validation score', linewidth=2)
    ax.fill_between(gamma_range, test_scores.mean(axis=1) - test_scores.std(axis=1),
                    test_scores.mean(axis=1) + test_scores.std(axis=1),
                    alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('gamma', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Validation Curve for gamma\n(C=10 fixed)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.01)
    
    # Add annotations
    ax.annotate('Underfitting', xy=(0.0001, 0.92), fontsize=10, ha='center', color='gray')
    ax.annotate('Overfitting', xy=(5, 0.92), fontsize=10, ha='center', color='gray')
    
    plt.tight_layout()
    plt.savefig('validation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: validation_curves.png")


# ============================================================================
# PART 7: PRACTICAL TIPS
# ============================================================================

def show_practical_tips():
    """
    Show practical tips for hyperparameter tuning
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PRACTICAL TIPS FOR HYPERPARAMETER TUNING                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. START WITH A COARSE GRID                                                 ║
║     ────────────────────────                                                 ║
║     • Use logarithmic spacing: [0.001, 0.01, 0.1, 1, 10, 100]               ║
║     • Find the general region where performance is good                     ║
║     • Then refine with a finer grid in that region                          ║
║                                                                              ║
║  2. USE APPROPRIATE CV STRATEGY                                              ║
║     ────────────────────────────                                             ║
║     • StratifiedKFold: For classification (maintains class balance)         ║
║     • KFold: For regression                                                  ║
║     • TimeSeriesSplit: For time series data                                  ║
║     • Typically use k=5 or k=10 folds                                        ║
║                                                                              ║
║  3. WATCH FOR OVERFITTING                                                    ║
║     ──────────────────────                                                   ║
║     • If train score >> validation score → overfitting                      ║
║     • Solution: Increase regularization (lower C, lower gamma)              ║
║                                                                              ║
║  4. WATCH FOR UNDERFITTING                                                   ║
║     ────────────────────────                                                 ║
║     • If both train and validation scores are low → underfitting            ║
║     • Solution: Decrease regularization (higher C, higher gamma)            ║
║                                                                              ║
║  5. CONSIDER ALTERNATIVES TO GRID SEARCH                                     ║
║     ─────────────────────────────────────                                    ║
║     • RandomizedSearchCV: Faster, good for many hyperparameters             ║
║     • Bayesian Optimization: More efficient sampling                        ║
║     • Halving Grid Search: Eliminates poor candidates early                 ║
║                                                                              ║
║  6. ALWAYS KEEP A HELD-OUT TEST SET                                          ║
║     ────────────────────────────────                                         ║
║     • Grid search optimizes on validation folds                             ║
║     • Final evaluation must be on unseen test data                          ║
║     • Never use test data during hyperparameter tuning!                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================================
# PART 8: COMPARISON OF SEARCH STRATEGIES
# ============================================================================

def compare_search_strategies(X, y):
    """
    Compare Grid Search vs Random Search
    """
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, loguniform
    import time
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              GRID SEARCH vs RANDOM SEARCH COMPARISON                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Grid Search
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }
    
    print("Grid Search:")
    print(f"  Parameters: {len(param_grid['C'])} x {len(param_grid['gamma'])} = "
          f"{len(param_grid['C']) * len(param_grid['gamma'])} combinations")
    
    start_time = time.time()
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    grid_time = time.time() - start_time
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best score: {grid_search.best_score_:.4f}")
    print(f"  Time: {grid_time:.2f}s")
    
    # Random Search (same number of iterations)
    param_dist = {
        'C': loguniform(0.01, 100),
        'gamma': loguniform(0.001, 10)
    }
    
    print("\nRandom Search (25 iterations):")
    
    start_time = time.time()
    random_search = RandomizedSearchCV(SVC(kernel='rbf'), param_dist, 
                                       n_iter=25, cv=5, n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    random_time = time.time() - start_time
    
    print(f"  Best params: C={random_search.best_params_['C']:.4f}, "
          f"gamma={random_search.best_params_['gamma']:.4f}")
    print(f"  Best score: {random_search.best_score_:.4f}")
    print(f"  Time: {random_time:.2f}s")
    
    print("""
    
KEY DIFFERENCES:
────────────────
Grid Search:
  ✓ Exhaustive - tries ALL combinations
  ✓ Guaranteed to find best in the grid
  ✗ Computationally expensive for large grids
  ✗ May miss optimal values between grid points

Random Search:
  ✓ More efficient for high-dimensional spaces
  ✓ Can explore continuous parameter ranges
  ✓ Often finds good solutions with fewer iterations
  ✗ Not guaranteed to find the absolute best
    """)
    
    return grid_search, random_search


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   COMPREHENSIVE GUIDE TO GRID SEARCH CROSS-VALIDATION")
    print("=" * 80)
    
    # Load and prepare data
    print("\nLoading Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.bincount(y)} (0=malignant, 1=benign)")
    
    # Run all explanations and demonstrations
    explain_cross_validation()
    explain_grid_search()
    explain_hyperparameters()
    
    # Step-by-step grid search
    results, best_params = grid_search_from_scratch_detailed(X_scaled, y)
    
    # Visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    visualize_grid_search_results(X_scaled, y)
    visualize_hyperparameter_effects(X_scaled, y)
    create_validation_curves(X_scaled, y)
    
    # Search strategy comparison
    compare_search_strategies(X_scaled, y)
    
    # Practical tips
    show_practical_tips()
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE!")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  • cv_visualization.png - Cross-validation fold diagram")
    print("  • grid_search_analysis.png - Grid search results analysis")
    print("  • hyperparameter_effects.png - Decision boundary changes")
    print("  • validation_curves.png - Train vs validation curves")