from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Create app
app = Flask(__name__)

# Preprocessing function (same as used in training)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news = request.form["news"]
        
        # 🔥 Clean the user input before transforming
        cleaned_news = clean_text(news)
        data = vectorizer.transform([cleaned_news])
        prediction = model.predict(data)[0]

        if prediction.lower() == "fake":
            result = "Fake News 🟥"
        else:
            result = "Real News 🟩"

        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)










