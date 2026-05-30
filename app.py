import streamlit as st
import torch
import torch.nn as nn

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# Config
# ==========================================

vocab_size = 10000
max_len = 200

# ==========================================
# Word Index
# ==========================================

word_index = imdb.get_word_index()

word_to_id = {
    k: (v + 3)
    for k, v in word_index.items()
}

word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# ==========================================
# Encode Function
# ==========================================

def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:

        if word in word_to_id:
            encoded.append(word_to_id[word])
        else:
            encoded.append(2)

    return encoded

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
# Load Model
# ==========================================

model = RNNModel()

model.load_state_dict(
    torch.load(
        "models/imdb_rnn_model.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()

# ==========================================
# Streamlit UI
# ==========================================

st.title("IMDb Sentiment Analysis using RNN")

review = st.text_area(
    "Enter Movie Review"
)

if st.button("Predict"):

    encoded_review = encode_review(review)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=max_len
    )

    input_tensor = torch.tensor(
        padded_review,
        dtype=torch.long
    )

    with torch.no_grad():

        prediction = model(input_tensor)

        score = prediction.item()

    st.write(f"Sentiment Score: {score:.4f}")

    if score >= 0.5:
        st.success("Positive Review")
    else:
        st.error("Negative Review")

