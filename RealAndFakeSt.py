import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

uploaded_true = st.file_uploader("Upload True.csv", type="csv")
uploaded_fake = st.file_uploader("Upload Fake.csv", type="csv")

if uploaded_true is not None and uploaded_fake is not None:
    df_true = pd.read_csv(uploaded_true)
    df_fake = pd.read_csv(uploaded_fake)

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

    # Calculate accuracy on training data
    y_train_pred = model.predict(x)
    training_accuracy = accuracy_score(y, y_train_pred)

    # Display accuracy
    st.write(f"Training Accuracy: {training_accuracy}")

    text = st.text_area("Enter the news text:")
    prediction = predict_fake_news(text)

    if prediction == 0:
        st.write("The news is Real")
    else:
        st.write("The news is Fake")
