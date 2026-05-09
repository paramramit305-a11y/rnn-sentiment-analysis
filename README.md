# 🎬 IMDb Sentiment Analysis using RNN

A deep learning project that classifies movie reviews as **Positive** or **Negative** using a PyTorch RNN model, deployed as an interactive web app with Streamlit.

---

## 🚀 Live Demo

> _[Add your Streamlit Cloud link here after deployment]_

---

## 📌 Project Overview

This project builds an end-to-end **Natural Language Processing (NLP)** pipeline to perform binary sentiment classification on IMDb movie reviews.

**Input:** A movie review (text)  
**Output:** Positive / Negative sentiment with confidence score

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch |
| NLP / Feature Extraction | TF-IDF Vectorizer (scikit-learn) |
| Model Architecture | Recurrent Neural Network (RNN) |
| Web App | Streamlit |
| Data | IMDb Movie Reviews Dataset |
| Language | Python 3.x |

---

## 🧠 How It Works

```
Raw Review Text
      ↓
Text Preprocessing (lowercasing, removing HTML tags, punctuation, stopwords)
      ↓
TF-IDF Vectorization (converts text to numerical features)
      ↓
RNN Model (PyTorch) — trained on IMDb dataset
      ↓
Sentiment Prediction + Confidence Score
```

**Key Design Choice:** TF-IDF + RNN combination — TF-IDF captures term importance while the RNN learns sequential patterns from the feature vectors.

---

## 📁 Project Structure

```
RNN/
├── models/
│   ├── rnn_sentiment_model.pt      # Trained PyTorch model weights
│   ├── tfidf_vectorizer.joblib     # Saved TF-IDF vectorizer
│   ├── label_mapping.json          # Label encoder mapping
│   └── model_metadata.json         # Model configuration info
├── src/
│   ├── config.py                   # Configuration constants
│   ├── model_service.py            # Prediction service layer
│   ├── preprocessing.py            # Text cleaning functions
│   └── rnn_model.py                # RNN architecture definition
├── app.py                          # Streamlit web app
├── RNN_for_sentiment_analysis.ipynb  # Training notebook
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Dataset | IMDb (50,000 reviews) |
| Task | Binary Classification (Positive / Negative) |
| Framework | PyTorch |

> _Add your accuracy/F1 score from the notebook here_

---

## 💡 Sample Predictions

| Review | Prediction | Confidence |
|---|---|---|
| "bad movie!" | Negative | 99.99% |
| "Absolutely loved it, masterpiece!" | Positive | ~99% |

---

## 📚 What I Learned

- Building and training RNN models from scratch using PyTorch
- NLP preprocessing pipeline (HTML tag removal, stopwords, text normalization)
- Combining TF-IDF feature extraction with deep learning models
- Saving and loading trained models with `torch.save` / `joblib`
- Deploying ML models as interactive web apps using Streamlit
- Structuring ML projects with modular `src/` architecture

---

## 🔗 Connect

**GitHub:** [paramramit305-a11y](https://github.com/paramramit305-a11y)  
**LinkedIn:** [Aman Banavali](https://linkedin.com/in/banavali-aman-97941a377)

---

> Built as part of my AI/ML learning journey — Prime AIML Batch by Shraddha Khapra
