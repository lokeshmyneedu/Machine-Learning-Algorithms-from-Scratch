import pandas as pd

# 1. Create a mock dataset: User, Movie, Rating
data = {
    'User_ID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'Movie': ['Star Wars', 'Empire Strikes Back', 'Titanic', 
              'Star Wars', 'Empire Strikes Back', 'The Notebook',
              'Titanic', 'The Notebook', 'Romeo & Juliet',
              'Star Wars', 'Titanic', 'Romeo & Juliet'],
    'Rating': [5, 5, 1, 4, 5, 2, 1, 4, 5, 5, 2, 4]
}

df = pd.DataFrame(data)

# 2. Create a Pivot Table (Rows = Users, Columns = Movies)
# This is our User-Item Matrix
movie_matrix = df.pivot_table(index='User_ID', columns='Movie', values='Rating')

# 3. Calculate Correlation between movies
# We use Pearson Correlation (similar to Adjusted Cosine)
# It finds movies that have similar rating patterns
movie_correlations = movie_matrix.corr(method='pearson')

# 4. Look at the "Similarity" to Star Wars
star_wars_similarities = movie_correlations['Star Wars'].sort_values(ascending=False)

print("Movies similar to 'Star Wars':")
print(star_wars_similarities)