# ==========================================
# IMDb Sentiment Analysis App using RNN
# ==========================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ==========================================
# Load Model
# ==========================================

model = load_model("models/imdb_rnn_model.keras")

# ==========================================
# IMDb Word Index
# ==========================================

word_index = imdb.get_word_index()

# ==========================================
# Prepare Dictionary
# ==========================================

word_to_id = {
    k: (v + 3)
    for k, v in word_index.items()
}

word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# ==========================================
# Encode Review Function
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
# Streamlit Front-End
# ==========================================

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    layout="centered"
)

st.title("IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to predict whether "
    "the sentiment is Positive or Negative."
)

review = st.text_area(
    "Enter Movie Review"
)

# ==========================================
# Prediction
# ==========================================

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        encoded_review = encode_review(review)

        padded_review = pad_sequences(
            [encoded_review],
            maxlen=200
        )

        prediction = model.predict(padded_review)

        sentiment_score = prediction[0][0]

        st.subheader("Prediction Result")

        st.write(
            f"Sentiment Score: {sentiment_score:.4f}"
        )

        if sentiment_score >= 0.5:

            st.success("Positive Review 😊")

        else:

            st.error("Negative Review 😔")