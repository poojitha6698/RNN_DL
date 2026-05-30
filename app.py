import streamlit as st
import torch
import torch.nn as nn
import joblib

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ==================================================
# Custom CSS
# ==================================================

st.markdown("""
<style>

.main{
background:#f5f7fa;
}

.title{
font-size:42px;
font-weight:bold;
text-align:center;
color:white;
padding:20px;
border-radius:15px;
background:linear-gradient(
90deg,
#4facfe,
#00f2fe
);
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 3px 12px rgba(0,0,0,0.2);
}

.big{
font-size:28px;
font-weight:bold;
}

</style>
""",
unsafe_allow_html=True)

# ==================================================
# Title
# ==================================================

st.markdown(
"""
<div class="title">
🎬 IMDb Movie Review Sentiment Analysis
</div>
""",
unsafe_allow_html=True
)

st.write("")

# ==================================================
# Load Tokenizer
# ==================================================

vectorizer = joblib.load(
    "models/tokenizer.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

# ==================================================
# Model
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

model = RNNModel()

model.load_state_dict(
    torch.load(
        "models/imdb_rnn_model.pth",
        map_location="cpu"
    )
)

model.eval()

# ==================================================
# User Input
# ==================================================

review = st.text_area(
    "Enter Movie Review",
    height=200
)

# ==================================================
# Prediction
# ==================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    vector = vectorizer.transform(
        [review]
    ).toarray()

    tensor = torch.tensor(
        vector,
        dtype=torch.float32
    )

    with torch.no_grad():

        prediction = model(
            tensor
        )

        score = prediction.item()

    st.write("---")

    if score >= 0.5:

        st.success(
            f"😊 Positive Review ({score:.2%})"
        )

        st.balloons()

    else:

        st.error(
            f"😔 Negative Review ({(1-score):.2%})"
        )

    st.progress(score)