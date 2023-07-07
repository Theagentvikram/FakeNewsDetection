import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk

nltk.download('stopwords')
stop_words = stopwords.words('english')

def preprocess_text(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_fake_news(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Streamlit App
st.title("Fake News Detection")

uploaded_file = st.file_uploader("Upload a dataset file (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess the data
    df['content'] = df['title'] + ' ' + df['text'] + ' ' + df['date']
    df['content'] = df['content'].apply(preprocess_text)

    # Train the model
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df['content'])
    y = df['label'].values
    model = LogisticRegression()
    model.fit(x, y)

    text = st.text_area("Enter the news text:")
    prediction = predict_fake_news(text)

    if prediction == 0:
        st.write("The news is Real")
    else:
        st.write("The news is Fake")
