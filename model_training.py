import os
import joblib
import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from torch.utils.data import TensorDataset, DataLoader

# ==================================================
# Create Models Folder
# ==================================================

os.makedirs("models", exist_ok=True)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("dataset/IMDB Dataset.csv")

# ==================================================
# Encode Labels
# ==================================================

label_encoder = LabelEncoder()

df["sentiment"] = label_encoder.fit_transform(
    df["sentiment"]
)

# ==================================================
# Text Vectorization
# ==================================================

vectorizer = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

X = vectorizer.fit_transform(
    df["review"]
).toarray()

y = df["sentiment"].values

# ==================================================
# Save Tokenizer & Encoder
# ==================================================

joblib.dump(
    vectorizer,
    "models/tokenizer.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

# ==================================================
# Train Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# Convert to Tensor
# ==================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

# ==================================================
# DataLoader
# ==================================================

dataset = TensorDataset(
    X_train,
    y_train
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

# ==================================================
# RNN Model
# ==================================================

class RNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=5000,
            hidden_size=128,
            batch_first=True
        )

        self.fc = nn.Linear(
            128,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.unsqueeze(1)

        output, hidden = self.rnn(x)

        out = self.fc(
            hidden.squeeze(0)
        )

        return self.sigmoid(out)

# ==================================================
# Train Model
# ==================================================

model = RNNModel()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    for X_batch, y_batch in loader:

        optimizer.zero_grad()

        outputs = model(
            X_batch
        ).squeeze()

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch+1}/{epochs} Loss:{loss.item():.4f}"
    )

# ==================================================
# Save Model
# ==================================================

torch.save(
    model.state_dict(),
    "models/imdb_rnn_model.pth"
)

print("Model Saved Successfully")