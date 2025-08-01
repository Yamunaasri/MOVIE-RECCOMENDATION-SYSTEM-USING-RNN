# MOVIE-RECCOMENDATION-SYSTEM-USING-RNN
# 🎬 Movie Review Sentiment Analysis with Simple RNN

This project is a complete end-to-end deep learning pipeline for **sentiment analysis** on movie reviews using a **Simple Recurrent Neural Network (RNN)**. It includes everything from text preprocessing to training, evaluation, saving the model, and deploying a **Streamlit** web app.

---

## 📌 Project Overview

- **Dataset**: IMDb-style reviews (synthetically generated for demonstration)
- **Goal**: Predict whether a review is **Positive** or **Negative**
- **Model**: Multi-layer Simple RNN using PyTorch
- **Deployment**: Streamlit-based web app for live predictions

---

## 🧠 Features

- Clean and preprocess text data  
- Build vocabulary and tokenize input  
- Define and train a multi-layer Simple RNN  
- Evaluate model performance  
- Save model and preprocessing pipeline  
- Streamlit app for user review sentiment prediction

---

## 🚀 How to Run Locally

### 1️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python rnn_training.py
```

This will:
- Train the model on synthetic IMDb reviews
- Save the model to `sentiment_rnn_model.pth`
- Save the preprocessor to `text_preprocessor.pkl`
- Generate a training performance plot (`training_history.png`)

### 3️⃣ Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

You’ll see a web interface where you can:
- Enter a review
- Click “Analyze Sentiment”
- View the result with confidence visuals

---

## 💾 Files Included

| File | Description |
|------|-------------|
| `rnn_training.py` | Main script for training and saving the model |
| `streamlit_app.py` | Streamlit app for live predictions |
| `requirements.txt` | Python packages needed |
| `README.md` | Project documentation |
| `sentiment_rnn_model.pth` | Trained PyTorch model (generated after training) |
| `text_preprocessor.pkl` | Saved preprocessing pipeline |

---

## 🌐 Optional: Deploy to the Cloud

You can deploy the app for free using [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your files to a public GitHub repo  
2. Connect the repo on Streamlit Cloud  
3. Set the startup command: `streamlit run streamlit_app.py`  
4. Done! Share your app with the world

<img width="1908" height="1004" alt="Screenshot 2025-08-01 144226" src="https://github.com/user-attachments/assets/33bd86f8-7b3a-43ef-b5c0-ee722af6ad4f" />

---

## ⚠️ Notes

- The dataset here is synthetic for simplicity. For real-world performance, consider the official [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).  
- No external pretrained embeddings (e.g., GloVe/BERT) are used.  
- Current model architecture is a simple RNN and can be upgraded to LSTM/GRU/Transformer for better results.


---

## 📎 License

This project is provided for educational purposes only.
