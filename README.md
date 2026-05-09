# 🎬 RNN Sentiment Analysis — IMDb Movie Reviews

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An end-to-end Deep Learning web application that classifies movie reviews as **Positive** or **Negative** using a PyTorch RNN model with TF-IDF feature extraction — deployed live on Streamlit Cloud.

🔗 **Live App:** [rnn-sentiment-analysis-4hsztw5c8vmuisxlyxxpqs.streamlit.app](https://rnn-sentiment-analysis-4hsztw5c8vmuisxlyxxpqs.streamlit.app)

---

## 📸 Preview

| Input | Prediction |
|-------|-----------|
| "bad movie!" | ❌ Negative — 99.99% confidence |
| "good movie" | ✅ Positive — 94.13% confidence |

---

## 🎯 Project Overview

This project builds a complete **NLP + Deep Learning pipeline** for binary sentiment classification on IMDb movie reviews. A user enters any movie review (or opinion text), and the app instantly predicts whether the sentiment is positive or negative — along with a confidence score and probability breakdown.

**Input:** Any movie review or opinion text  
**Output:** Positive / Negative sentiment with confidence score + probability scores

---

## 🚀 Features

- **Real-time sentiment prediction** with confidence percentage
- **Probability scores** for both Positive and Negative classes
- **Text preprocessing viewer** — see how the model cleaned your input
- **Modular src/ architecture** — clean, production-style code structure
- **Deployed on Streamlit Cloud** — accessible from any browser

---

## 🧠 ML Pipeline

### Dataset
- **IMDb Movie Reviews Dataset** — 50,000 reviews (25K positive, 25K negative)
- Balanced binary classification task

### Preprocessing
- Lowercasing all text
- Removing HTML tags (common in IMDb data)
- Removing punctuation and special characters
- Stopword removal
- Clean text passed to TF-IDF vectorizer

### Feature Extraction
- **TF-IDF Vectorizer** (scikit-learn) — converts text to numerical feature vectors
- Captures term importance across the corpus

### Model Architecture
- **Recurrent Neural Network (RNN)** built with PyTorch
- Input: TF-IDF feature vector
- Hidden layers with RNN units
- Output: 2-class softmax (Positive / Negative)

### Key Design Choice
TF-IDF + RNN combination — TF-IDF handles vocabulary-level feature importance while the RNN learns patterns from the feature sequence. This hybrid approach gives strong performance without requiring word embeddings.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Dataset | IMDb (50,000 reviews) |
| Task | Binary Classification |
| Model | PyTorch RNN |
| Feature Extraction | TF-IDF Vectorizer |

> Sample: "bad movie!" → **Negative 99.99%** | "good movie" → **Positive 94.13%**

---

## 💡 Sample Predictions

| Review | Prediction | Confidence |
|--------|-----------|------------|
| "bad movie!" | Negative | 99.99% |
| "good movie" | Positive | 94.13% |
| "Absolutely loved it, masterpiece!" | Positive | ~99% |
| "Worst film I have ever seen" | Negative | ~99% |

---

## 🗂️ Project Structure

```
rnn-sentiment-analysis/
│
├── models/
│   ├── rnn_sentiment_model.pt      # Trained PyTorch model weights (2.5 MB)
│   ├── tfidf_vectorizer.joblib     # Saved TF-IDF vectorizer (179 KB)
│   ├── label_mapping.json          # Label encoder mapping
│   └── model_metadata.json         # Model configuration info
│
├── src/
│   ├── config.py                   # Configuration constants & paths
│   ├── model_service.py            # Prediction service layer
│   ├── preprocessing.py            # Text cleaning functions
│   └── rnn_model.py                # RNN architecture definition
│
├── app.py                          # Streamlit web app
├── RNN_for_sentiment_analysis.ipynb  # Training notebook (full pipeline)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/paramramit305-a11y/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Dependencies

```
streamlit==1.45.1
scikit-learn==1.6.1
joblib==1.4.2
pandas==2.2.3
numpy==2.2.5
torch==2.5.1
python-dotenv==1.0.1
openai==1.78.1
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| PyTorch | RNN model training & inference |
| Scikit-learn | TF-IDF vectorization |
| Pandas & NumPy | Data processing |
| Joblib | Model serialization |
| Streamlit | Web app framework & deployment |
| GitHub | Version control |
| Streamlit Cloud | Free hosting & deployment |

---

## 📚 What I Learned

- Building and training RNN models from scratch using PyTorch
- NLP preprocessing pipeline (HTML tag removal, stopwords, text normalization)
- Combining TF-IDF feature extraction with deep learning models
- Saving and loading trained models with `torch.save` / `joblib`
- Deploying ML models as interactive web apps using Streamlit
- Structuring ML projects with modular `src/` production-style architecture

---

## 👤 Author

**Parmar Amit**
- GitHub: [@paramramit305-a11y](https://github.com/paramramit305-a11y)
- LinkedIn: [Aman Banavali](https://www.linkedin.com/in/parmar-amit-97941a377)

---
