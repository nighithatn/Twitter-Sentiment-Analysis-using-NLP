# 🐦 Twitter Sentiment Analysis using BiLSTM & FastAPI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-green)

---

## 📌 Project Overview

This project implements an end-to-end **Twitter Sentiment Analysis system** using:

- 🧠 Bidirectional LSTM (BiLSTM)
- 📊 NLP preprocessing
- ⚖️ Class imbalance handling
- 🚀 FastAPI deployment

The system classifies tweets into:

- Negative
- Neutral
- Positive

---

## 🎯 Aim

To build, evaluate, and deploy a deep learning model that performs sentiment classification on Twitter text data.

---

## 🗂 Dataset

- Twitter sentiment dataset
- Columns: `text`, `sentiment`
- 3-class classification problem

---

## 🔄 Workflow

1. Dataset loading
2. Text preprocessing
3. Label encoding
4. Train-test split
5. Tokenization & padding
6. BiLSTM model building
7. Training with class weights
8. Evaluation (Accuracy, Precision, Recall, F1)
9. Model saving
10. FastAPI deployment

---

## 🧹 Text Preprocessing

- Lowercasing
- URL removal
- Mention & hashtag removal
- Special character removal
- Stopword removal
- Lemmatization

---

## 🧠 Model Architecture

- Embedding Layer
- Bidirectional LSTM
- Dropout
- Dense + Softmax output

Loss Function: `sparse_categorical_crossentropy`  
Optimizer: `Adam`

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | ~69% |
| Macro F1 | 0.69 |
| Weighted F1 | 0.69 |

- Positive sentiment predicted best
- Neutral class most challenging
- Balanced class performance

---

## 🚀 FastAPI Deployment

The trained model is deployed using FastAPI.

### Run the API:

```bash
cd Api
uvicorn app:app --reload
```

### Open in browser:

```
http://127.0.0.1:8000/docs
```

### Example API Request:

```json
{
  "text": "I love this project!"
}
```

---

## 📁 Project Structure

```
Twitter Sentiment Analysis using NLP/
app.py
dataLoader.py
preprocessing.py
tokenizerUtils.py
modelBuilder.py
trainer.py
evaluator.py
saveUtils.py
main.py
requirements.txt
```

---

## 🔮 Future Improvements

- Pretrained embeddings (GloVe, FastText)
- Transformer models (BERT)
- Cloud deployment
- Frontend UI integration

---

## 👩‍💻 Author

Nighitha T. N.

---


## ⭐ If you found this project useful, give it a star!
