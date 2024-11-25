import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# read in the MovieLens 100K dataset
df = pd.read_csv("data/ml-latest-small/ratings.csv")

# Map user_id and item_id to integer indices
user_mapping = {user: idx for idx, user in enumerate(df["userId"].unique())}
item_mapping = {item: idx for idx, item in enumerate(df["movieId"].unique())}
df["userId"] = df["userId"].map(user_mapping)
df["movieId"] = df["movieId"].map(item_mapping)

# For implicit feedback, we treat all ratings as positive interactions
df["interaction"] = (df["rating"] > 0).astype(float)  # Binary implicit feedback


# Split the data into training and testing
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Create a PyTorch Dataset class
class ImplicitFeedbackDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data["userId"].values, dtype=torch.long) # Convert to tensor
        self.items = torch.tensor(data["movieId"].values, dtype=torch.long) # Convert to tensor
        self.labels = torch.tensor(data["interaction"].values, dtype=torch.float) # Convert to tensor

    def __len__(self):
        return len(self.labels) # Number of samples

    def __getitem__(self, idx): # Return a sample
        return self.users[idx], self.items[idx], self.labels[idx]


# Create DataLoaders for training and testing
train_dataset = ImplicitFeedbackDataset(train) # Create dataset
test_dataset = ImplicitFeedbackDataset(test) # Create dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=32) # Create DataLoader

# Define the recommendation model
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


# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if GPU is available
num_users = len(user_mapping) # Number of users
num_items = len(item_mapping) # Number of items

model = ImplicitFeedbackModel(num_users, num_items).to(device) # Move model to device
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam optimizer

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    total_loss = 0 # Initialize total loss
    for user_ids, item_ids, labels in train_loader:
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device) # Move to device

        # Forward pass
        predictions = model(user_ids, item_ids).squeeze() # Predict interactions
        loss = criterion(predictions, labels) # Compute loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluate the model
model.eval()
hits, total = 0, 0
with torch.no_grad():
    for user_ids, item_ids, labels in test_loader:
        user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
        predictions = model(user_ids, item_ids).squeeze()
        predictions = (predictions > 0.5).float()  # Convert probabilities to binary predictions
        hits += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = hits / total
print(f"Test Accuracy: {accuracy:.4f}")

# Generate recommendations for a sample user
sample_user_id = 0  # Example user
all_items = torch.arange(num_items).to(device)
sample_user = torch.tensor([sample_user_id] * num_items).to(device)

with torch.no_grad():
    scores = model(sample_user, all_items).squeeze()
    recommended_items = torch.argsort(scores, descending=True)[:5]  # Top 5 recommendations

# Reverse the mapping from indices back to item IDs
reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}

# Get the original item IDs for the recommended indices
recommended_item_ids = [reverse_item_mapping[idx] for idx in recommended_items.cpu().numpy()]

print(f"Recommended items for user {sample_user_id}: {recommended_item_ids}")
