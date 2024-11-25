import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset
df = pd.read_csv("data/ml-latest-small/ratings.csv")
print("Sample data:")
print(df.head())

# Columns: userId, movieId, rating, timestamp
# We focus on `userId`, `movieId`, and `rating`
# Map userId and movieId to consecutive integers
user_mapping = {user: idx for idx, user in enumerate(df["userId"].unique())}
movie_mapping = {movie: idx for idx, movie in enumerate(df["movieId"].unique())}

df["userId"] = df["userId"].map(user_mapping)
df["movieId"] = df["movieId"].map(movie_mapping)

# Split the data into training and testing
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Create a PyTorch Dataset class
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data["userId"].values, dtype=torch.long)
        self.movies = torch.tensor(data["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(data["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


# Create DataLoaders for training and testing
train_dataset = MovieLensDataset(train)
test_dataset = MovieLensDataset(test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the recommendation model
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


# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_users = len(user_mapping)
num_movies = len(movie_mapping)

model = ExplicitFeedbackModel(num_users, num_movies).to(device)
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for user_ids, movie_ids, ratings in train_loader:
        user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)

        # Forward pass
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluate the model
model.eval()
total_loss = 0
with torch.no_grad():
    for user_ids, movie_ids, ratings in test_loader:
        user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)
        total_loss += loss.item()

print(f"Test Loss (MSE): {total_loss / len(test_loader):.4f}")

# Generate predictions for a sample user
sample_user_id = 0  # Example user
all_movies = torch.arange(num_movies).to(device)
sample_user = torch.tensor([sample_user_id] * num_movies).to(device)

with torch.no_grad():
    predicted_ratings = model(sample_user, all_movies)
    recommended_movie_indices = torch.argsort(predicted_ratings, descending=True)[:5]  # Top 5 recommendations

# Map recommendations back to original movie IDs
reverse_movie_mapping = {idx: movie for movie, idx in movie_mapping.items()}
recommended_movie_ids = [reverse_movie_mapping[idx] for idx in recommended_movie_indices.cpu().numpy()]

print(f"Top recommendations for user {sample_user_id}: {recommended_movie_ids}")
