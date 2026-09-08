# 🧠 Next Word Prediction using LSTM RNN

An end-to-end Natural Language Processing (NLP) deep learning project that predicts the next word in a sequence of text using a Long Short-Term Memory (LSTM) Recurrent Neural Network. The model is trained on William Shakespeare's *Hamlet* and deployed as an interactive web application built with **Streamlit**.

---

## 📌 Project Overview

Next Word Prediction is a fundamental task in NLP with practical applications in autocomplete systems, predictive keyboards, search engines, and assistive writing tools.

This project covers:
- **Data Collection & Cleaning**: Sourced Shakespeare's *Hamlet* via NLTK Gutenberg corpus.
- **Text Preprocessing**: Tokenization, n-gram sequence generation, and sequence padding.
- **Model Development**: Multi-layer LSTM architecture with embedding and regularization (Dropout).
- **Interactive UI**: Streamlit web interface for real-time word prediction with custom styling.

---

## 🏗️ Model Architecture

The neural network is built using Keras & TensorFlow:

| Layer | Type | Output Shape | Parameters | Details |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Embedding** | `(None, 13, 100)` | 481,800 | Vocabulary size: 4,818, Embedding dimension: 100 |
| 2 | **LSTM** | `(None, 13, 150)` | 150,600 | 150 units with `return_sequences=True` |
| 3 | **Dropout** | `(None, 13, 150)` | 0 | Dropout rate: 0.2 |
| 4 | **LSTM** | `(None, 100)` | 100,400 | 100 units |
| 5 | **Dense** | `(None, 4818)` | 486,618 | Softmax activation across vocabulary |

- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** `adam`
- **Metrics:** `accuracy`
- **Callbacks:** `EarlyStopping` monitoring validation loss

---

## 📂 Repository Structure

```plaintext
Next-Word-Prediction-main/
│
├── LSTM RNN/
│   ├── app.py                  # Streamlit web application
│   ├── experiments.ipynb       # Data exploration, model building & training notebook
│   ├── hamlet.txt              # Raw Shakespeare's Hamlet text corpus
│   ├── next_word_lstm.h5       # Pretrained LSTM model file
│   └── tokenizer.pickle        # Fitted Keras Tokenizer
│
├── nextword.jpg                # Background image for the Streamlit UI
├── requirements.txt            # Python dependencies
├── LICENSE                     # License information
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11 (Recommended for TensorFlow 2.15 compatibility)
- [Anaconda / Miniconda](https://www.anaconda.com/) or standard Python `venv`

---

### Installation

#### Option A: Using Conda (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Next-Word-Prediction.git
cd Next-Word-Prediction

# 2. Create and activate a Python 3.10 environment
conda create -n nextword python=3.10 -y
conda activate nextword

# 3. Install required packages
pip install -r requirements.txt
```

#### Option B: Using Python `venv`

```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Windows (Git Bash)
source ./venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

> **Note:** Run the command from the root directory of the repository so relative file paths resolve correctly.

```bash
streamlit run "LSTM RNN/app.py"
```

Once running, access the web app in your browser at:
**`http://localhost:8501`**

---

## 🔬 Training & Experiments

To inspect the training pipeline, data preprocessing steps, or retrain the model with custom parameters:

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open [`LSTM RNN/experiments.ipynb`](file:///d:/AIML/projects/next-word-pred/Next-Word-Prediction-main/Next-Word-Prediction-main/LSTM%20RNN/experiments.ipynb).
3. Run the notebook cells to re-download the Gutenberg corpus, prepare the sequences, train the model, and export updated `.h5` and `.pickle` files.

---

## 🛠️ Tech Stack

- **Deep Learning Framework:** TensorFlow / Keras
- **NLP & Preprocessing:** NLTK, Scikit-Learn
- **Web App / UI:** Streamlit, HTML5, Custom CSS
- **Data Manipulation & Visualization:** NumPy, Pandas, Matplotlib, Seaborn

---

## 📄 License

This project is licensed under the terms described in the [LICENSE](file:///d:/AIML/projects/next-word-pred/Next-Word-Prediction-main/Next-Word-Prediction-main/LICENSE) file.