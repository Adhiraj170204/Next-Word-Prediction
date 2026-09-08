import streamlit as st
import numpy as np
import pickle
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('LSTM RNN/next_word_lstm.h5')
with open('LSTM RNN/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# Set page configuration
st.set_page_config(page_title="Next Word Predictor", layout="centered")

# Set background using local image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("nextword.jpg")

# Custom CSS for UI styling
st.markdown("""
    <style>
    .title-box {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #2193b0;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title container
st.markdown("""
<div class='title-box'>
    <h1>🧠 Next Word Prediction</h1>
    <p> Enter a sequence of words and get the next predicted word using a trained LSTM model</p>
</div>
""", unsafe_allow_html=True)

# Input and prediction
input_text = st.text_input("🔡 Input your text sequence:", "to be or not to be")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.success(f"**Predicted Next Word:** `{next_word}`")

# Optional: Add a copyright footer
st.markdown("""
<hr style="margin-top: 3rem; border-top: 1px solid #bbb;">
<div style='text-align: center; color: white; font-size: 14px;'>
    © 2025 Next Word Predictor | Developed using Streamlit & LSTM
</div>
""", unsafe_allow_html=True)
