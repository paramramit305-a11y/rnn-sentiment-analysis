<div align="center">

# 🎬 RNN Sentiment Analysis — IMDb Movie Reviews

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rnn-sentiment-analysis-4hsztw5c8vmuisxlyxxpqs.streamlit.app)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> An end-to-end Deep Learning web application that classifies movie reviews as **Positive** or **Negative** using a PyTorch RNN model with TF-IDF feature extraction — deployed live on Streamlit Cloud.

🔗 **[Live App](https://rnn-sentiment-analysis-4hsztw5c8vmuisxlyxxpqs.streamlit.app)**

</div>

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

## 🔮 Future Improvements

- [ ] Replace TF-IDF with Word Embeddings (GloVe / Word2Vec)
- [ ] Upgrade to LSTM / Bi-LSTM for better sequence modeling
- [ ] Fine-tune BERT for state-of-the-art sentiment classification
- [ ] Add multi-class sentiment (Very Positive, Neutral, Very Negative)
- [ ] REST API with FastAPI for programmatic access

---

## 👤 Author

<div align="center">

**Parmar Amit**
*BSc IT (AIML) | Gokul Global University*

[![GitHub](https://img.shields.io/badge/GitHub-paramramit305--a11y-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/paramramit305-a11y)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Parmar%20Amit-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parmar-amit-97941a377)

</div>

---

<div align="center">

⭐ If you find this project useful, please consider giving it a star!

</div>
