import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data: Movies and their tags
movies = pd.DataFrame({
    'movie_id': [1, 2, 3],
    'title': ['John Wick', 'The Notebook', 'Die Hard'],
    'content': ['Action Thriller Assassin', 'Romance Drama Love', 'Action Explosion Terrorist']
})

# 2. Vectorize the 'content' using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# 3. Assume the User likes 'John Wick' (index 0)
# We calculate similarity of all movies to John Wick
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

# 4. Show results
movies['similarity_score'] = cosine_sim.flatten()
print(movies.sort_values(by='similarity_score', ascending=False))