# ==========================================
# IMDb Sentiment Analysis using RNN (PyTorch)
# ==========================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# Create Models Folder
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# ==========================================
# Convert to Tensors
# ==========================================

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==========================================
# DataLoader
# ==========================================

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# ==========================================
# RNN Model
# ==========================================

class RNNModel(nn.Module):

    def __init__(self):

        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            32
        )

        self.rnn = nn.RNN(
            32,
            32,
            batch_first=True
        )

        self.fc1 = nn.Linear(32, 16)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.rnn(x)

        hidden = hidden.squeeze(0)

        x = self.fc1(hidden)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x

# ==========================================
# Initialize Model
# ==========================================

model = RNNModel()

criterion = nn.BCELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# Training
# ==========================================

epochs = 3

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ==========================================
# Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "models/imdb_rnn_model.pth"
)

print("Model Saved Successfully!")

