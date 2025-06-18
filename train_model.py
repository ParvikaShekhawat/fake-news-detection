# ============================
# Step 1: train_model.py
# ============================

import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed)

# Load and merge datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")
fake_df['label'] = "FAKE"  # FAKE
true_df['label'] = "REAL" # REAL

# Merge and shuffle
combined_df = pd.concat([fake_df, true_df], ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Preprocess
combined_df['text'] = combined_df['text'].apply(clean_text)# loop

# Split
X_train, X_test, y_train, y_test = train_test_split(combined_df['text'], combined_df['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)

# Evaluate
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))



