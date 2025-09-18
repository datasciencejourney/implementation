from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and tokenizer
model = load_model("C:\\Users\\user\\Documents\\Minor Project\\implementation\\LSTM\\lstm_English_30Epochs.keras")

with open("C:\\Users\\user\\Documents\\Minor Project\\implementation\\LSTM\\lstmEnglishTokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_config = f.read()
tokenizer = tokenizer_from_json(tokenizer_config)

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)
    words = [lemmatizer.lemmatize(word.lower()) for word in text.split() if word.isalpha() and word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Clean and process the text
        cleaned_text = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=128, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        probability = float(prediction[0][0])
        is_ai_generated = "AI Generated" if probability > 0.5 else "Human Written"
        confidence = probability if probability > 0.5 else 1 - probability
        
        return jsonify({
            'prediction': is_ai_generated,
            'confidence': f"{confidence * 100:.2f}%",
            'original_text': text
        })

if __name__ == '__main__':
    app.run(debug=True)