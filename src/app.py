import os
import re
import pickle
from flask import Flask, jsonify, request

# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import contractions

# Initialize Flask app
app = Flask(__name__)

# Define file paths for model loading
base_dir = os.path.dirname(os.path.dirname(__file__))
model_dir = os.path.join(base_dir, 'models')

# Load the vocabulary and sentiment analysis model using pickle
with open(os.path.join(model_dir, 'vectorizers.pkl'), 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)

with open(os.path.join(model_dir, 'sentiment_model.pkl'), 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text (remove stopwords, lemmatize)
def clean_tweet(tweet, stop_words=False):
    new_tweet = re.sub(r"'s\b", " is", tweet)
    new_tweet = re.sub("#", "", new_tweet)
    new_tweet = re.sub("@[A-Za-z0-9]+", "", new_tweet)
    new_tweet = re.sub(r"http\S+", "", new_tweet)
    new_tweet = contractions.fix(new_tweet)
    new_tweet = re.sub(r"[^a-zA-Z]", " ", new_tweet)
    new_tweet = new_tweet.lower().strip()
    new_tweet = new_tweet.replace('`', "'")

    # Download NLTK data
    corpus = nltk.download('punkt')
    nltk.download('stopwords')

    if stop_words == True:
        # write logic to remove stop words
        pass

    # Tokenize the tweet and apply lemmatization
    words = word_tokenize(new_tweet)
    cleaned_tweet = ' '.join(lemmatizer.lemmatize(word) for word in words)

    return cleaned_tweet


@app.route('/')
def index():
    response = "twitter sentiment analysis..."
    return jsonify({"response": response})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.get_json()
        
        # Assuming the JSON data contains a key called 'tweet'
        input_tweet = input_data.get('tweet')

        if input_tweet is None:
            return jsonify({'error': 'Missing or invalid JSON data'}), 400

        # Preprocess the input tweet
        transformed_tweet = clean_tweet(input_tweet)

        # Vectorize the tweet using the initialized TF-IDF vectorizer
        vector_input = vocab.transform([transformed_tweet])

        # Make a prediction using the loaded sentiment analysis model
        result = model.predict(vector_input)[0]

        # Map the result to sentiment classes
        if result == 0:
            prediction = 'Negative'
        elif result == 1:
            prediction = 'Neutral'
        else:
            prediction = 'Positive'

        return jsonify({'prediction': prediction})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)