import base64
import json
import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "next_word_model.keras"
TOKENIZER_PATH = ROOT / "models" / "tokenizer.pickle"
METRICS_PATH = ROOT / "results" / "metrics.json"
BACKGROUND_PATH = ROOT / "nextword.jpg"
TOP_K = 5

st.set_page_config(page_title="Next Word Predictor", layout="centered")


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    with open(METRICS_PATH) as handle:
        metrics = json.load(handle)
    return model, tokenizer, metrics


def predict_top_k(model, tokenizer, max_sequence_len, text, k=TOP_K):
    # Tokenise exactly like the training notebook: lowercase + the saved word regex.
    words = re.findall(tokenizer["token_pattern"], text.lower())
    word_index, oov_id, pad_id = tokenizer["word_index"], tokenizer["oov_id"], tokenizer["pad_id"]
    input_len = max_sequence_len - 1
    ids = [word_index.get(w, oov_id) for w in words][-input_len:]
    x = np.array([[pad_id] * (input_len - len(ids)) + ids], dtype=np.int32)
    probs = model.predict(x, verbose=0)[0].copy()
    probs[[pad_id, oov_id]] = 0  # never suggest padding or <OOV>
    top = np.argsort(-probs)[:k]
    return [(tokenizer["index_word"][int(i)], float(probs[i])) for i in top]


# Set background using local image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        body, .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background(BACKGROUND_PATH)

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
    .suggestions {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .suggestion-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0; }
    .suggestion-word { width: 7rem; font-weight: bold; font-family: monospace; font-size: 16px; }
    .suggestion-bar { height: 10px; background-color: #2193b0; border-radius: 4px; }
    .suggestion-prob { font-size: 14px; color: #ddd; }
    </style>
""", unsafe_allow_html=True)

# Title container
st.markdown("""
<div class='title-box'>
    <h1>🧠 Next Word Prediction</h1>
    <p>Enter a sequence of words and get the top-5 next-word suggestions from a recurrent model trained on Shakespeare</p>
</div>
""", unsafe_allow_html=True)

missing = [p.relative_to(ROOT).as_posix() for p in (MODEL_PATH, TOKENIZER_PATH, METRICS_PATH) if not p.exists()]
if missing:
    st.error(
        "Missing trained artifacts: " + ", ".join(missing) + ". Run "
        "`notebooks/next_word_prediction.ipynb` and unzip its output into the repository root."
    )
    st.stop()

model, tokenizer, metrics = load_artifacts()
max_sequence_len = metrics["max_sequence_len"]

# Input and prediction
input_text = st.text_input("🔡 Input your text sequence:", "to be or not to")

if st.button("Suggest Next Words"):
    if not re.findall(tokenizer["token_pattern"], input_text.lower()):
        st.warning("Please enter at least one word.")
    else:
        suggestions = predict_top_k(model, tokenizer, max_sequence_len, input_text)
        top_prob = suggestions[0][1] or 1.0
        rows = "".join(
            f"<div class='suggestion-row'>"
            f"<span class='suggestion-word'>{word}</span>"
            f"<div class='suggestion-bar' style='width:{max(prob / top_prob, 0.02) * 60:.1f}%'></div>"
            f"<span class='suggestion-prob'>{prob:.1%}</span>"
            f"</div>"
            for word, prob in suggestions
        )
        st.markdown(f"<div class='suggestions'><b>Top-{TOP_K} suggestions</b>{rows}</div>",
                    unsafe_allow_html=True)

st.caption(f"Model: {metrics.get('selected_model', 'RNN')} · vocabulary: {metrics['vocab_size']:,} words")

# Optional: Add a copyright footer
st.markdown("""
<hr style="margin-top: 3rem; border-top: 1px solid #bbb;">
<div style='text-align: center; color: white; font-size: 14px;'>
    © 2025 Next Word Predictor | Developed using Streamlit, LSTM & GRU
</div>
""", unsafe_allow_html=True)
