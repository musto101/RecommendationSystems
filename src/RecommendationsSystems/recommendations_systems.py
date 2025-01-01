# create a wrapper function to allow for the use of different models
import torch
import torch.nn as nn
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k


class ImplicitFeedbackModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim) # User embeddings
        self.item_embeddings = nn.Embedding(num_items, embedding_dim) # Item embeddings
        self.output_layer = nn.Linear(embedding_dim * 2, 1)  # Combine user and item embeddings

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        item_embeds = self.item_embeddings(item_ids)  # [batch_size, embedding_dim]
        combined = torch.cat([user_embeds, item_embeds], dim=1)  # [batch_size, embedding_dim * 2]
        return torch.sigmoid(self.output_layer(combined))  # Predict interaction probability

class ExplicitFeedbackModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)  # Fully connected layer 1
        self.fc2 = nn.Linear(64, 32)  # Fully connected layer 2
        self.fc3 = nn.Linear(32, 1)  # Output layer to predict the rating

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        movie_embeds = self.movie_embeddings(movie_ids)  # [batch_size, embedding_dim]
        combined = torch.cat([user_embeds, movie_embeds], dim=1)  # [batch_size, embedding_dim * 2]
        x = torch.relu(self.fc1(combined))  # Hidden layer 1
        x = torch.relu(self.fc2(x))  # Hidden layer 2
        return self.fc3(x).squeeze()  # Output predicted rating. squeeze() removes the last dimension


class RecommendationSystem:

    def train_model(model_type, train_matrix, test_matrix, movie_features_matrix):

        user_mapping = {user: idx for idx, user in enumerate(train_matrix["userId"].unique())}
        item_mapping = {item: idx for idx, item in enumerate(train_matrix["movieId"].unique())}
        train_matrix["userId"] = train_matrix["userId"].map(user_mapping)
        train_matrix["movieId"] = train_matrix["movieId"].map(item_mapping)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_users = len(user_mapping)
        num_movies = len(item_mapping)

        if model_type == 'implicit_feedback':
            model = ImplicitFeedbackModel(num_users, num_movies).to(device)
        elif model_type == 'explicit_feedback':
            model = ExplicitFeedbackModel(num_users, num_movies).to(device)
        elif model_type == 'hybrid':
            model = LightFM(loss="warp", no_components=32)
            model.fit(train_matrix, item_features=movie_features_matrix, epochs=10,
                      num_threads=4, verbose=True,)

        # evaluate the model