# 📰 Fake News Detection Web App

A simple Flask-based web application that uses machine learning to detect whether a piece of news is **Fake** or **Real**.

---

## 🚀 Features

- Detects fake vs. real news using a trained Logistic Regression model
- Built using Python, Flask, Scikit-learn, and NLTK
- Clean and user-friendly web interface
- Preprocessing includes lowercasing, stemming, stopword removal, and punctuation cleanup

---

## 🧠 Machine Learning Model

- **Model:** Logistic Regression
- **Text Vectorization:** TF-IDF (max features = 5000)
- **Preprocessing:** NLTK stopword removal and Porter stemming
- **Training Dataset:** Combined `Fake.csv` and `True.csv` news datasets

---

## 🛠️ How to Run

1. **Clone this repo**  
   `git clone https://github.com/ParvikaShekhawat/fake-news-detector.git`

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the app**  
`python app.py`

4. **Open in your browser**  
   After running the app, go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)


## 🧠 Model Info

- Trained on [Kaggle Fake/Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Vectorized with TF-IDF (max_features=5000)
- Built with Logistic Regression

## 📂 Project Structure


## 🧪 Example Inputs

- ✅ Real: "NASA's Artemis mission aims to return humans to the Moon by 2026."
- ❌ Fake: "Aliens have landed in Australia and taken over Parliament."

## 👨‍💻 Author

**Your Name**  
GitHub: [@ParvikaShekhawat](https://github.com/yourusername)

## 📜 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for more details.



