import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')
st.set_page_config(page_title="Fake News Detection", page_icon="IMG.png")

def preprocess_text(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_fake_news(text, vectorizer, model):
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

    # Looking and replacing null data
    df_true = df_true.fillna('')
    df_fake = df_fake.fillna('')

    # Merging author name and news title for easier use
    df_true['content'] = df_true['title'] + ' ' + df_true['text'] + ' ' + df_true['date']
    df_fake['content'] = df_fake['title'] + ' ' + df_fake['text'] + ' ' + df_fake['date']

    # Preprocess the data
    df_true['content'] = df_true['content'].apply(preprocess_text)
    df_fake['content'] = df_fake['content'].apply(preprocess_text)

    # Combine the datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)

    x = df['content'].values
    y = df['label'].values

    # Converting the textual data to numerical data for easier use
    vectorizer = TfidfVectorizer()
    vectorizer.fit(x)
    x = vectorizer.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    x_train_pred = model.predict(x_train)
    training_data_acc = accuracy_score(x_train_pred, y_train)

    x_test_pred = model.predict(x_test)
    test_data_acc = accuracy_score(x_test_pred, y_test)

    st.write(f"Training Accuracy: {training_data_acc}")
    st.write(f"Test Accuracy: {test_data_acc}")

    text = st.text_area("Enter the news text:")
    
    if st.button("Predict"):
        if text:
            with st.spinner("Predicting..."):
                prediction = predict_fake_news(text)

            if prediction == 0:
                st.write("The news is Real")
            else:
                st.write("The news is Fake")
