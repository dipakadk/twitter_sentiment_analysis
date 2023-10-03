from flask import Flask, render_template, request
import pickle
import re
import contractions
import nltk

# Download the 'punkt' corpus
corpus=nltk.download('punkt')
# Download the 'stopwords' corpus
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3050)
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizers.pkl', 'rb'))

# large_corpus=[corpus]
# tfidf.fit_transform(large_corpus)

#smallcorp='Ah! Now I have done Philosophy,\nI have finished Law and Medicine,\nAnd sadly even Theology:\nTaken fierce pains, from end to end.\nNow here I am, a fool for sure!\nNo wiser than I was before:'


def text_cleaner_without_stopwords(text):
    new_text = re.sub(r"'s\b", " is", text)
    new_text = re.sub("#", "", new_text)
    new_text = re.sub("@[A-Za-z0-9]+", "", new_text)
    new_text = re.sub(r"http\S+", "", new_text)
    new_text = contractions.fix(new_text)
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = new_text.lower().strip()
    new_text = new_text.replace('`', "'")

    cleaned_text = ''
    for token in new_text.split():
        cleaned_text = cleaned_text + lemmatizer.lemmatize(token) + ' '

    return " ".join(cleaned_text)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        input_text = request.form.get('text')
        transformed_text = text_cleaner_without_stopwords(input_text)
        vector_input = tfidf.fit_transform([transformed_text])
        result = model.predict(vector_input)[0]

        prediction = ""
        if result == 0:
            prediction = 'Negative'
        elif result == 1:
            prediction = 'Neutral'
        else:
            prediction = 'Positive'

        return f'Predicted class: {prediction}'

if __name__ == '__main__':
    app.run(debug=True)
