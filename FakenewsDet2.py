import streamlit as st
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

nltk.download('stopwords')
stop_words = stopwords.words('english')
port_stem = PorterStemmer()

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

# Streamlit App
st.title("Fake News Detection")

uploaded_true = st.file_uploader("Upload True.csv", type="csv")
uploaded_fake = st.file_uploader("Upload Fake.csv", type="csv")

if uploaded_true is not None and uploaded_fake is not None:
    df_true = pd.read_csv(uploaded_true)
    df_fake = pd.read_csv(uploaded_fake)

    # Preprocess the data
    df_true['content'] = df_true['title'] + ' ' + df_true['text'] + ' ' + df_true['subject'] + ' ' + df_true['date']
    df_true['content'] = df_true['content'].apply(preprocess_text)

    df_fake['content'] = df_fake['title'] + ' ' + df_fake['text'] + ' ' + df_fake['subject'] + ' ' + df_fake['date']
    df_fake['content'] = df_fake['content'].apply(preprocess_text)

    # Combine the datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Separate the data and label
    X = df['content'].values
    y = np.concatenate([np.zeros(len(df_true)), np.ones(len(df_fake))])

    # Split the dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    y_train_pred = model.predict(X_train_vectorized)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test_vectorized)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # User input
    text = st.text_area("Enter the news text:")
    if text:
        prediction = predict_fake_news(text)
        if prediction == 0:
            st.write("The news is Real")
        else:
            st.write("The news is Fake")

    # Display the accuracies
    st.write("Training Accuracy:", train_accuracy)
    st.write("Testing Accuracy:", test_accuracy)
