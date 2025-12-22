import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# 1. Create a dummy dataset (User, Item, Rating)
data_dict = {
    'user': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie'],
    'item': ['Matrix', 'Titanic', 'Matrix', 'Inception', 'Titanic', 'Inception'],
    'rating': [5, 1, 4, 5, 2, 4]
}
df = pd.DataFrame(data_dict)

# 2. Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

# 3. Use SVD (Singular Value Decomposition) - a Matrix Factorization algorithm
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# 4. Predict Alice's rating for 'Inception' (which she hasn't seen)
prediction = algo.predict('Alice', 'Inception')
print(f"Predicted rating for Alice on Inception: {prediction.est:.2f}")