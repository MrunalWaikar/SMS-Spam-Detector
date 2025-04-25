# SMS-Spam-Detector
# 📩 SMS Spam Classifier

An intelligent machine learning model that classifies SMS messages as **spam** or **ham** (non-spam) using Natural Language Processing (NLP). This project demonstrates text classification using feature extraction techniques and the Multinomial Naive Bayes algorithm.

---

## 🧠 About the Project

Spam messages clutter our inboxes and pose security risks. This project aims to:

- Clean and preprocess SMS data using NLP.
- Transform text into meaningful numerical features.
- Train a classifier to accurately detect spam messages.
- Evaluate the model using standard performance metrics.

The final model is capable of identifying spam messages with high precision and accuracy.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size:** 5,574 SMS messages
- **Labels:** `ham` (non-spam), `spam`

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn
- TF-IDF Vectorizer
- Jupyter Notebook

---

## 🧰 Features

- Text preprocessing: punctuation removal, stopwords filtering, stemming
- Feature extraction with **TF-IDF Vectorization**
- Training with **Multinomial Naive Bayes**
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## 🚀 Getting Started

### ✅ Prerequisites

Ensure Python 3.x is installed with the following packages:

```bash
pip install pandas numpy nltk scikit-learn
