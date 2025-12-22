import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP DATA ---
movies = pd.DataFrame({
    'movie_id': [0, 1, 2, 3],
    'title': ['Toy Story', 'Akira', 'The Lion King', 'Ghost in the Shell'],
    'content': ['Animation Kids Disney', 'Anime Sci-Fi Cyberpunk', 'Animation Kids Musical', 'Anime Sci-Fi Cyborg']
})

# --- 2. CONTENT-BASED COMPONENT (TF-IDF) ---
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['content'])
content_sim = cosine_similarity(tfidf_matrix)

def get_content_score(movie_idx):
    # Returns similarity of all movies to the movie at movie_idx
    return content_sim[movie_idx]

# --- 3. COLLABORATIVE COMPONENT (Simulated Matrix Factorization) ---
# Rows: Users, Columns: Movies (Scores 0-5)
# Let's say we have 3 users and 4 movies
user_item_matrix = np.array([
    [5, 0, 4, 1], # User 0 likes Disney, dislikes Anime
    [1, 5, 0, 5], # User 1 likes Anime, dislikes Disney
    [0, 0, 0, 0]  # User 2 is new (Cold Start)
])

def get_collaborative_score(user_id):
    # In a real app, you'd use SVD here. 
    # For this example, we'll return the raw ratings of the user
    return user_item_matrix[user_id]

# --- 4. THE HYBRID FUNCTION ---
def hybrid_recommender(user_id, last_watched_movie_idx, alpha=0.5):
    """
    alpha: weight for Content (0 to 1). 
    (1-alpha) will be the weight for Collaborative.
    """
    c_scores = get_content_score(last_watched_movie_idx)
    cf_scores = get_collaborative_score(user_id)
    
    # Normalize CF scores to be between 0 and 1 (matching Cosine Similarity)
    cf_scores_norm = cf_scores / 5.0 
    
    # Calculate Weighted Average
    hybrid_scores = (alpha * c_scores) + ((1 - alpha) * cf_scores_norm)
    
    movies['final_score'] = hybrid_scores
    return movies.sort_values(by='final_score', ascending=False)

# Example: User 0 (Disney fan) just watched 'Toy Story' (index 0)
print("Recommendations for User 0:")
print(hybrid_recommender(user_id=0, last_watched_movie_idx=0, alpha=0.5))