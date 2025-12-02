"""
Comprehensive Implementation of Bernoulli and Multinomial Models
with Real-World Examples
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
sns.set_style("whitegrid")

print("="*80)
print("BERNOULLI MODEL - DETAILED EXPLANATION AND IMPLEMENTATION")
print("="*80)

# ============================================================================
# PART 1: BERNOULLI DISTRIBUTION - THEORETICAL EXAMPLE
# ============================================================================

print("\n1. BERNOULLI DISTRIBUTION - BASIC THEORY")
print("-" * 80)

# Parameter
p_success = 0.7  # Probability of success (e.g., probability an email is spam)

# Generate samples
n_samples = 1000
bernoulli_samples = np.random.binomial(n=1, p=p_success, size=n_samples)

print(f"Probability of success (p): {p_success}")
print(f"Probability of failure (1-p): {1-p_success}")
print(f"\nGenerated {n_samples} samples:")
print(f"Number of successes (1s): {np.sum(bernoulli_samples)}")
print(f"Number of failures (0s): {n_samples - np.sum(bernoulli_samples)}")
print(f"Empirical probability: {np.mean(bernoulli_samples):.3f}")
print(f"Theoretical mean: {p_success}")
print(f"Theoretical variance: {p_success * (1-p_success):.3f}")
print(f"Empirical variance: {np.var(bernoulli_samples):.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart of outcomes
unique, counts = np.unique(bernoulli_samples, return_counts=True)
axes[0].bar(unique, counts/n_samples, color=['red', 'green'], alpha=0.7, edgecolor='black')
axes[0].axhline(y=p_success, color='blue', linestyle='--', label=f'Theoretical p={p_success}')
axes[0].set_xlabel('Outcome', fontsize=12)
axes[0].set_ylabel('Probability', fontsize=12)
axes[0].set_title('Bernoulli Distribution\n(Email Spam Classification)', fontsize=14, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Not Spam (0)', 'Spam (1)'])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Cumulative distribution
cumsum = np.cumsum(bernoulli_samples) / np.arange(1, n_samples + 1)
axes[1].plot(cumsum, linewidth=2)
axes[1].axhline(y=p_success, color='red', linestyle='--', label=f'True p={p_success}')
axes[1].set_xlabel('Number of Trials', fontsize=12)
axes[1].set_ylabel('Cumulative Average', fontsize=12)
axes[1].set_title('Law of Large Numbers\n(Convergence to True Probability)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
print("\n✓ Bernoulli distribution plot saved!")

# ============================================================================
# PART 2: REAL-WORLD EXAMPLE - EMAIL SPAM CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("2. REAL-WORLD EXAMPLE: EMAIL SPAM CLASSIFICATION (BERNOULLI NAIVE BAYES)")
print("="*80)

# Create synthetic email dataset
def create_email_dataset(n_samples=500):
    """Create synthetic email dataset with binary features"""
    
    # Features (presence/absence of words)
    features = ['contains_free', 'contains_winner', 'contains_click', 
                'contains_urgent', 'contains_money', 'has_link',
                'all_caps_subject', 'multiple_exclamation']
    
    data = []
    labels = []
    
    # Generate spam emails (label = 1)
    for _ in range(n_samples // 2):
        # Spam emails have higher probability of spam indicators
        email = [np.random.choice([0, 1], p=[0.2, 0.8]) for _ in range(len(features))]
        data.append(email)
        labels.append(1)
    
    # Generate legitimate emails (label = 0)
    for _ in range(n_samples // 2):
        # Legitimate emails have lower probability of spam indicators
        email = [np.random.choice([0, 1], p=[0.8, 0.2]) for _ in range(len(features))]
        data.append(email)
        labels.append(0)
    
    return np.array(data), np.array(labels), features

# Generate dataset
X, y, feature_names = create_email_dataset(n_samples=1000)

print(f"\nDataset created:")
print(f"Total emails: {len(X)}")
print(f"Spam emails: {np.sum(y)}")
print(f"Legitimate emails: {len(y) - np.sum(y)}")
print(f"Number of features: {X.shape[1]}")
print(f"\nFeatures (binary - presence/absence):")
for i, feature in enumerate(feature_names):
    print(f"  {i+1}. {feature}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTrain set: {len(X_train)} emails")
print(f"Test set: {len(X_test)} emails")

# Train Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train, y_train)

# Make predictions
y_pred = bernoulli_nb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n" + "="*50)
print(f"MODEL PERFORMANCE")
print(f"="*50)
print(f"Accuracy: {accuracy:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam']))

# Feature probabilities
print("\n" + "="*50)
print("LEARNED PARAMETERS")
print("="*50)
print("\nProbability of each feature given class:")
print(f"\n{'Feature':<25} {'P(feature|Spam)':<20} {'P(feature|Legit)':<20}")
print("-" * 65)

for i, feature in enumerate(feature_names):
    prob_spam = np.exp(bernoulli_nb.feature_log_prob_[1][i])
    prob_legit = np.exp(bernoulli_nb.feature_log_prob_[0][i])
    print(f"{feature:<25} {prob_spam:<20.3f} {prob_legit:<20.3f}")

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Spam'],
            yticklabels=['Legitimate', 'Spam'],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - Email Spam Classification\n(Bernoulli Naive Bayes)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
print("\n✓ Confusion matrix saved!")

# ============================================================================
# PART 3: MULTINOMIAL DISTRIBUTION - THEORETICAL EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("MULTINOMIAL MODEL - DETAILED EXPLANATION AND IMPLEMENTATION")
print("="*80)

print("\n3. MULTINOMIAL DISTRIBUTION - BASIC THEORY")
print("-" * 80)

# Parameters
categories = ['Sports', 'Politics', 'Technology', 'Entertainment']
probabilities = [0.3, 0.25, 0.25, 0.2]  # Must sum to 1
n_trials = 100  # Number of articles

print(f"Categories: {categories}")
print(f"Probabilities: {probabilities}")
print(f"Number of trials: {n_trials}")

# Generate samples
multinomial_samples = np.random.multinomial(n_trials, probabilities, size=1000)

print(f"\nGenerated 1000 experiments, each with {n_trials} trials")
print(f"\nAverage counts per category:")
mean_counts = np.mean(multinomial_samples, axis=0)
for i, cat in enumerate(categories):
    theoretical = n_trials * probabilities[i]
    print(f"  {cat}: {mean_counts[i]:.1f} (theoretical: {theoretical:.1f})")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Expected distribution
axes[0, 0].bar(categories, [p * n_trials for p in probabilities], 
               color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Expected Count', fontsize=11)
axes[0, 0].set_title('Theoretical Distribution\n(Expected Counts per Category)', 
                     fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Actual distribution from one sample
sample_idx = 0
axes[0, 1].bar(categories, multinomial_samples[sample_idx], 
               color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Observed Count', fontsize=11)
axes[0, 1].set_title('Single Experiment\n(One Draw from Multinomial)', 
                     fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Distribution of counts for Sports category
axes[1, 0].hist(multinomial_samples[:, 0], bins=20, 
                color='green', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=n_trials * probabilities[0], color='red', 
                   linestyle='--', linewidth=2, label='Expected value')
axes[1, 0].set_xlabel('Count', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title(f'Distribution of {categories[0]} Counts\n(Across 1000 Experiments)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Comparison of empirical vs theoretical
axes[1, 1].bar(np.arange(len(categories)) - 0.2, mean_counts, 0.4, 
               label='Empirical', color='purple', alpha=0.7, edgecolor='black')
axes[1, 1].bar(np.arange(len(categories)) + 0.2, [p * n_trials for p in probabilities], 0.4,
               label='Theoretical', color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(np.arange(len(categories)))
axes[1, 1].set_xticklabels(categories, rotation=45)
axes[1, 1].set_ylabel('Average Count', fontsize=11)
axes[1, 1].set_title('Empirical vs Theoretical\n(Average Across 1000 Experiments)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
print("\n✓ Multinomial distribution plot saved!")

# ============================================================================
# PART 4: REAL-WORLD EXAMPLE - DOCUMENT CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("4. REAL-WORLD EXAMPLE: DOCUMENT CLASSIFICATION (MULTINOMIAL NAIVE BAYES)")
print("="*80)

# Create synthetic document dataset
documents = [
    # Sports documents
    "football game match player score goal team win league championship",
    "basketball court dunk three pointer team score player game",
    "soccer match goal player team tournament world cup championship",
    "tennis match serve player game set tournament grand slam",
    "baseball game home run pitch player team score inning",
    "football playoff game team player score touchdown championship",
    "basketball finals game player score team championship trophy",
    "soccer world cup game team player goal match tournament",
    "tennis wimbledon player serve game match championship",
    "baseball world series game team player pitch score",
    
    # Politics documents
    "president election vote government policy senate congress bill",
    "congress debate bill law vote senator representative legislation",
    "election campaign vote candidate party president governor debate",
    "government policy reform bill senate house congress legislation",
    "president white house policy reform government congress bill",
    "senate vote legislation bill law congress representative policy",
    "election primary vote candidate campaign party debate",
    "congress bill policy reform law government senate vote",
    "president administration policy government reform congress bill",
    "legislation vote senate congress bill law policy reform",
    
    # Technology documents
    "software programming code computer algorithm developer application system",
    "artificial intelligence machine learning algorithm model neural network data",
    "computer hardware processor chip system technology performance speed",
    "software development code programming language application framework system",
    "data science algorithm model analysis machine learning python",
    "cloud computing server technology infrastructure application data system",
    "programming language code software developer application algorithm system",
    "machine learning model algorithm neural network data training",
    "computer technology system hardware software application processor",
    "software engineer code programming development application system algorithm",
    
    # Entertainment documents
    "movie film actor director cinema hollywood box office award",
    "music concert artist album song performance tour festival",
    "television show series episode actor drama comedy network",
    "movie premiere actor film director hollywood cinema award",
    "concert performance music artist tour stage show festival",
    "television series episode show actor network drama season",
    "film festival movie director cinema award actor hollywood",
    "music album artist song performance concert tour recording",
    "show television episode series actor drama network season",
    "movie actor film director cinema award hollywood premiere"
]

labels = (
    ['Sports'] * 10 + 
    ['Politics'] * 10 + 
    ['Technology'] * 10 + 
    ['Entertainment'] * 10
)

print(f"\nDocument dataset created:")
print(f"Total documents: {len(documents)}")
print(f"Categories: {list(set(labels))}")
for label in set(labels):
    count = labels.count(label)
    print(f"  {label}: {count} documents")

# Convert documents to word count features
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(documents)

print(f"\nFeature extraction:")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature matrix shape: {X_counts.shape}")

# Split dataset
X_train_doc, X_test_doc, y_train_doc, y_test_doc = train_test_split(
    X_counts, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"\nTrain set: {X_train_doc.shape[0]} documents")
print(f"Test set: {X_test_doc.shape[0]} documents")

# Train Multinomial Naive Bayes
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_doc, y_train_doc)

# Make predictions
y_pred_doc = multinomial_nb.predict(X_test_doc)

# Evaluate
accuracy_doc = accuracy_score(y_test_doc, y_pred_doc)
print(f"\n" + "="*50)
print(f"MODEL PERFORMANCE")
print(f"="*50)
print(f"Accuracy: {accuracy_doc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test_doc, y_pred_doc))

# Show most informative words for each category
print("\n" + "="*50)
print("MOST INFORMATIVE WORDS PER CATEGORY")
print("="*50)

feature_names_vocab = vectorizer.get_feature_names_out()
n_top_words = 8

for idx, category in enumerate(multinomial_nb.classes_):
    print(f"\n{category}:")
    top_indices = np.argsort(multinomial_nb.feature_log_prob_[idx])[-n_top_words:][::-1]
    top_words = [feature_names_vocab[i] for i in top_indices]
    print(f"  {', '.join(top_words)}")

# Test with new documents
print("\n" + "="*50)
print("TESTING WITH NEW DOCUMENTS")
print("="*50)

test_docs = [
    "The basketball team won the championship with an amazing final game",
    "The president signed a new policy bill in congress yesterday",
    "New machine learning algorithm improves artificial intelligence performance",
    "The movie won best picture award at the cinema festival"
]

test_vectors = vectorizer.transform(test_docs)
predictions = multinomial_nb.predict(test_vectors)
probabilities = multinomial_nb.predict_proba(test_vectors)

for i, doc in enumerate(test_docs):
    print(f"\nDocument: '{doc[:60]}...'")
    print(f"Predicted category: {predictions[i]}")
    print(f"Confidence scores:")
    for j, category in enumerate(multinomial_nb.classes_):
        print(f"  {category}: {probabilities[i][j]:.3f}")

# Visualize confusion matrix
cm_doc = confusion_matrix(y_test_doc, y_pred_doc, labels=multinomial_nb.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_doc, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=multinomial_nb.classes_,
            yticklabels=multinomial_nb.classes_,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Category', fontsize=12)
plt.ylabel('True Category', fontsize=12)
plt.title('Confusion Matrix - Document Classification\n(Multinomial Naive Bayes)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
print("\n✓ Confusion matrix saved!")

# ============================================================================
# PART 5: COMPARISON AND KEY DIFFERENCES
# ============================================================================

print("\n" + "="*80)
print("5. KEY DIFFERENCES: BERNOULLI vs MULTINOMIAL")
print("="*80)

comparison_data = {
    'Aspect': [
        'Outcome Type',
        'Number of Categories',
        'Feature Representation',
        'Use Case',
        'Example',
        'Parameters'
    ],
    'Bernoulli': [
        'Binary (0 or 1)',
        '2 categories',
        'Binary (presence/absence)',
        'Binary classification',
        'Spam detection',
        'Single probability p'
    ],
    'Multinomial': [
        'Count-based',
        'Multiple categories (k > 2)',
        'Frequency counts',
        'Multi-class classification',
        'Topic classification',
        'Probability vector (p1, p2, ..., pk)'
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
BERNOULLI MODEL:
- Used for binary outcomes (yes/no, present/absent)
- Features are binary (0 or 1)
- Best for: Spam detection, presence/absence of features
- Example: Does an email contain the word "free"? Yes(1) or No(0)

MULTINOMIAL MODEL:
- Used for count data with multiple categories
- Features are frequency counts
- Best for: Text classification, document categorization
- Example: How many times does "sport" appear in a document? 0, 1, 2, 3...

Both are commonly used in Naive Bayes classifiers for text classification!
""")

print("\n" + "="*80)
print("All visualizations  d to /mnt/user-data/outputs/")
print("="*80)

plt.show()