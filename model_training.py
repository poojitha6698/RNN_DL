# ==========================================
# IMDb Sentiment Analysis using RNN
# Model Training File
# ==========================================

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# Create Models Folder
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Load IMDb Dataset
# ==========================================

vocab_size = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

# ==========================================
# Padding Sequences
# ==========================================

X_train = pad_sequences(
    X_train,
    maxlen=max_len
)

X_test = pad_sequences(
    X_test,
    maxlen=max_len
)

# ==========================================
# Build RNN Model
# ==========================================

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_len
    )
)

model.add(
    SimpleRNN(
        128,
        activation='tanh'
    )
)

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# Model Summary
# ==========================================

model.summary()

# ==========================================
# Early Stopping
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# ==========================================
# Evaluate Model
# ==========================================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy:.4f}")

# ==========================================
# Save Model
# ==========================================

model.save("models/imdb_rnn_model.h5")

print("\nModel Saved Successfully!")

# ==========================================
# Save Training History
# ==========================================

with open("models/training_history.pkl", "wb") as file:
    pickle.dump(history.history, file)

print("Training History Saved!")

# ==========================================
# Accuracy Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()

plt.savefig("models/accuracy_graph.png")

plt.show()

print("Accuracy Graph Saved!")