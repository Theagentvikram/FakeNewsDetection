import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import joblib as jb
nltk.download('stopwords')
stop_words = stopwords.words('english')
port_stem = PorterStemmer()

model=jb.load("finaldump.joblib")
vectorizer=jb.load("vect.dat")

def preprocess_text(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_fake_news(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

text = input("Enter the news text:")
if text:
    prediction = predict_fake_news(text)
    if prediction == 0:
        print("The news is Real")
    else:
        print("The news is Fake")

acc=jb.load("acc.dat")
print("Training Accuracy:", acc[0])
print("Testing Accuracy:", acc[1])