# Step 1: Import Libraries and Load the Model
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import streamlit as st

# Define the PyTorch model class
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        output = self.fc(rnn_out[:, -1, :])
        output = self.sigmoid(output)
        return output

# Load the IMDB dataset word index
@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

# Load model
@st.cache_resource
def load_pytorch_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters (must match training parameters)
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 128
    output_dim = 1
    
    model = SimpleRNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    
    try:
        model.load_state_dict(torch.load('simple_rnn_imdb_pytorch.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, device, True
    except FileNotFoundError:
        return None, device, False

# Helper Functions
def decode_review(encoded_review, reverse_word_index):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text, word_index, device):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    tensor_review = torch.LongTensor(padded_review).to(device)
    return tensor_review

def predict_sentiment(model, text, word_index, device):
    model.eval()
    with torch.no_grad():
        preprocessed_input = preprocess_text(text, word_index, device)
        prediction = model(preprocessed_input)
        prediction_score = prediction.cpu().numpy()[0][0]
        sentiment = 'Positive' if prediction_score > 0.5 else 'Negative'
        return sentiment, prediction_score

# Streamlit app
def main():
    st.title('IMDB Movie Review Sentiment Analysis')
    st.write('**PyTorch RNN Model** - Enter a movie review to classify it as positive or negative.')
    
    # Load data and model
    word_index, reverse_word_index = load_word_index()
    model, device, model_loaded = load_pytorch_model()
    
    if not model_loaded:
        st.error("Model file 'simple_rnn_imdb_pytorch.pth' not found!")
        st.error("Please train the model first using the simplernn.ipynb notebook.")
        st.stop()
    
    st.success(f"PyTorch model loaded successfully! Using device: {device}")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Device**: {device}")
        st.write(f"**Vocabulary Size**: 10,000")
        st.write(f"**Embedding Dim**: 128")
        st.write(f"**Hidden Dim**: 128")
        st.write(f"**Max Sequence Length**: 500")
        
        st.header("Example Reviews")
        examples = [
            "This movie was fantastic! Great acting and plot.",
            "Terrible film. Waste of time and money.",
            "Average movie, nothing special but okay.",
            "Brilliant cinematography and performances!"
        ]
        
        for example in examples:
            if st.button(f"Try: {example[:30]}...", key=example):
                st.session_state.example_text = example
    
    # User input
    default_text = st.session_state.get('example_text', '')
    user_input = st.text_area('Movie Review', value=default_text, height=100)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        predict_btn = st.button('Classify', type='primary')
    
    with col2:
        clear_btn = st.button('Clear')
        if clear_btn:
            st.session_state.example_text = ''
            st.rerun()
    
    if predict_btn and user_input.strip():
        with st.spinner('Analyzing sentiment...'):
            try:
                sentiment, score = predict_sentiment(model, user_input, word_index, device)
                
                # Display results
                st.write("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if sentiment == 'Positive':
                        st.success(f"**{sentiment}**")
                    else:
                        st.error(f"**{sentiment}**")
                
                with col2:
                    st.metric("Prediction Score", f"{score:.4f}")
                
                with col3:
                    confidence = abs(score - 0.5) * 200
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Progress bar for visualization
                st.write("**Sentiment Distribution:**")
                st.progress(score, text=f"Negative ← {score:.3f} → Positive")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                
    elif predict_btn:
        st.warning('Please enter a movie review to analyze.')
    
    # Instructions
    with st.expander("How to use"):
        st.write("""
        1. **Enter a movie review** in the text area above
        2. **Click 'Classify'** to get the sentiment prediction
        3. **View the results** showing sentiment, score, and confidence
        4. **Try the examples** from the sidebar for quick testing
        
        **Score interpretation:**
        - **0.0 - 0.5**: Negative sentiment
        - **0.5 - 1.0**: Positive sentiment
        - **Closer to 0 or 1**: Higher confidence
        """)

if __name__ == "__main__":
    main()
