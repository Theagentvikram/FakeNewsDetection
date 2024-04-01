import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
import streamlit as st

nltk.download('stopwords')

# Printing the stopwords in English
print(stopwords.words('english'))

# Loading the datasets to pandas DataFrames
df_true = pd.read_csv(r"C:\Users\abhic\Downloads\True.csv")
df_fake = pd.read_csv(r"C:\Users\abhic\Downloads\Fake.csv")

# Adding a label column to each DataFrame
df_true['label'] = 0
df_fake['label'] = 1

# Concatenating the datasets
df = pd.concat([df_true, df_fake], ignore_index=True)

# Merging the Title, Text, Subject, and Date columns
df['content'] = df['title'] + ' ' + df['text'] + ' ' + df['subject'] + ' ' + df['date']

# Preprocessing the content column
ps = PorterStemmer()
df['content'] = df['content'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['content'] = df['content'].apply(lambda x: x.lower())
df['content'] = df['content'].apply(lambda x: x.split())
df['content'] = df['content'].apply(lambda x: [ps.stem(word) for word in x if word not in stopwords.words('english')])
df['content'] = df['content'].apply(lambda x: ' '.join(x))

# Separating the data and label
X = df['content'].values
y = df['label'].values

# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting on the training data
y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", training_accuracy)

# Predicting on the test data
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Streamlit App
st.title("Fake News Detection")

text = st.text_area("Enter the news text:")
if text:
    preprocessed_text = re.sub('[^a-zA-Z]', ' ', text)
    preprocessed_text = preprocessed_text.lower()
    preprocessed_text = preprocessed_text.split()
    preprocessed_text = [ps.stem(word) for word in preprocessed_text if word not in stopwords.words('english')]
    preprocessed_text = ' '.join(preprocessed_text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]

    if prediction == 0:
        st.write("The news is Real")
    else:
        st.write("The news is Fake")
