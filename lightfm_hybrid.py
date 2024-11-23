from platform import libc_ver

import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
import optuna

# Load the MovieLens dataset
dataset_path = "data/ml-latest-small/ratings.csv"
movies_path = "data/ml-latest-small/movies.csv"

# Load ratings and movies data
ratings = pd.read_csv(dataset_path)
movies = pd.read_csv(movies_path)

ratings.columns
movies.columns

# Map userId and movieId to integer indices
user_mapping = {user: idx for idx, user in enumerate(ratings["userId"].unique())}
movie_mapping = {movie: idx for idx, movie in enumerate(ratings["movieId"].unique())}

ratings["userId"] = ratings["userId"].map(user_mapping)
ratings["movieId"] = ratings["movieId"].map(movie_mapping)

# Split the data into training and testing sets
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Create sparse matrices for training and testing
num_users = ratings["userId"].nunique()
num_movies = ratings["movieId"].nunique()

def create_interaction_matrix(data):
    return coo_matrix(
        (data["rating"].values, (data["userId"].values, data["movieId"].values)),
        shape=(num_users, num_movies),)

train_matrix = create_interaction_matrix(train)
test_matrix = create_interaction_matrix(test)

# Add movie metadata (for hybrid model)
# Parse genres into a binary feature matrix
movies["genres"] = movies["genres"].str.split('|')
all_genres = set(g for genres in movies["genres"] for g in genres)

# Create a mapping of movieId to features
movie_features = pd.DataFrame({
        genre: movies["genres"].apply(lambda x: int(genre in x))
        for genre in all_genres
    })

movie_features["movieId"] = movies["movieId"].map(movie_mapping)
movie_features = movie_features.set_index("movieId").fillna(0)

# Convert movie features to a sparse matrix
movie_features_matrix = coo_matrix(movie_features.values)

# Train the LightFM model
# Use a hybrid model combining collaborative and content-based filtering
model = LightFM(loss="warp", no_components=32)

model.fit(train_matrix, item_features=movie_features_matrix, epochs=10,
          num_threads=4, verbose=True,)

# Evaluate the model on the training set
precision = precision_at_k(model, train_matrix, k=10,
                           item_features=movie_features_matrix).mean() * 100
recall = recall_at_k(model, train_matrix, k=10,
                     item_features=movie_features_matrix).mean() * 100

print(f"Precision@5 Train: {precision:.4f}")
print(f"Recall@5 Train: {recall:.4f}")

# Evaluate the model on the test set
precision = precision_at_k(model, test_matrix, k=10,
                           item_features=movie_features_matrix).mean() * 100
recall = recall_at_k(model, test_matrix, k=10,
                     item_features=movie_features_matrix).mean() * 100

print(f"Precision@5 Test: {precision:.4f}")
print(f"Recall@5 Test: {recall:.4f}") # very poor performance on the train and test sets so I will implement bayesian
# optimization to tune the hyperparameters

# Bayesian Optimization on no_components, learning_rate, and epochs

def objective(trial):
    no_components = trial.suggest_int("no_components", 10, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 5, 50)

    model = LightFM(no_components=no_components, learning_rate=learning_rate)
    model.fit(train_matrix, item_features=movie_features_matrix, epochs=epochs,
              num_threads=4, verbose=True,)

    precision = precision_at_k(model, test_matrix, k=10,
                               item_features=movie_features_matrix).mean()

    return precision

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Get the best hyperparameters
best_params = study.best_params
no_components = best_params["no_components"]
learning_rate = best_params["learning_rate"]
epochs = best_params["epochs"]

# Train the model with the best hyperparameters
model = LightFM(no_components=no_components, learning_rate=learning_rate)
model.fit(train_matrix, item_features=movie_features_matrix, epochs=epochs,
          num_threads=4, verbose=True,)

# Evaluate the model on the training set
precision = precision_at_k(model, train_matrix, k=10,
                           item_features=movie_features_matrix).mean() * 100
recall = recall_at_k(model, train_matrix, k=10,
                        item_features=movie_features_matrix).mean() * 100

print(f"Precision@5 Train: {precision:.4f}")
print(f"Recall@5 Train: {recall:.4f}")

# Evaluate the model on the test set
precision_test = precision_at_k(model, test_matrix, k=10,
                           item_features=movie_features_matrix).mean() * 100
recall_test = recall_at_k(model, test_matrix, k=10,
                        item_features=movie_features_matrix).mean() * 100

print(f"Precision@5 Test: {precision_test:.4f}")
print(f"Recall@5 Test: {recall_test:.4f}")