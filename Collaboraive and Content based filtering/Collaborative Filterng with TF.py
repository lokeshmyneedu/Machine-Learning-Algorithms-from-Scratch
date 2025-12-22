import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import numpy as np

# 1. Prepare Mock Data
data = {
    'user_id': [0, 0, 1, 1, 2, 2, 3, 3],
    'movie_id': [0, 1, 0, 2, 1, 2, 0, 2],
    'rating': [5, 1, 4, 5, 2, 4, 5, 1]
}
df = pd.DataFrame(data)

num_users = df.user_id.nunique()
num_movies = df.movie_id.nunique()
embedding_size = 50 # The size of the latent factor vector

# 2. Build the Recommender Model
class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        # User embedding + bias
        self.user_embedding = layers.Embedding(num_users, embedding_size, 
                                               embeddings_initializer="he_normal",
                                               embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        
        # Movie embedding + bias
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, 
                                                embeddings_initializer="he_normal",
                                                embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        # The dot product represents the interaction between user and movie
        dot_product = tf.reduce_sum(tf.multiply(user_vector, movie_vector), axis=1, keepdims=True)
        
        # Add biases and pass through sigmoid (scaled to rating range)
        x = dot_product + user_bias + movie_bias
        return tf.nn.sigmoid(x) * 5  # Scaling sigmoid (0-1) to ratings (0-5)

# 3. Compile and Train
model = RecommenderNet(num_users, num_movies, embedding_size)
model.compile(loss="mse", optimizer="adam")

# Inputs: (user_id, movie_id) | Target: rating
x = df[['user_id', 'movie_id']].values
y = df['rating'].values.astype(np.float32)

model.fit(x=x, y=y, batch_size=64, epochs=5, verbose=1)

# 4. Make a Prediction
# Predict rating for User 0 on Movie 2
test_input = np.array([[0, 2]])
predicted_rating = model.predict(test_input)
print(f"Predicted Rating: {predicted_rating[0][0]:.2f}")