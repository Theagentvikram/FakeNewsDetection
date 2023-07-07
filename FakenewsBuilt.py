import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import requests

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

true_csv_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/True.csv"
fake_csv_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/Fake.csv"

true_csv_data = requests.get(true_csv_url).content
fake_csv_data = requests.get(fake_csv_url).content

df_true = pd.read_csv(io.StringIO(true_csv_data.decode('utf-8')))
df_fake = pd.read_csv(io.StringIO(fake_csv_data.decode('utf-8')))

# Preprocess the data
df_true['content'] = df_true['title'] + ' ' + df_true['text'] + ' ' + df_true['date']
df_true['content'] = df_true['content'].apply(preprocess_text)

df_fake['content'] = df_fake['title'] + ' ' + df_fake['text'] + ' ' + df_fake['date']
df_fake['content'] = df_fake['content'].apply(preprocess_text)

# Combine the datasets
df = pd.concat([df_true, df_fake], ignore_index=True)

# Train the model
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['content'])
y = np.concatenate([np.zeros(len(df_true)), np.ones(len(df_fake))])
model = LogisticRegression()
model.fit(x, y)

text = st.text_area("Enter the news text:")
prediction = predict_fake_news(text)

if prediction == 0:
    st.write("The news is Real")
else:
    st.write("The news is Fake")
