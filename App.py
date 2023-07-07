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

# Read the true news dataset
df_true = pd.read_csv(r"C:\Users\abhic\Downloads\true.csv")
df_true['label'] = 0  # Set label as 0 for true news

# Read the fake news dataset
df_fake = pd.read_csv(r'C:\Users\abhic\Downloads\fake.csv')
df_fake['label'] = 1  # Set label as 1 for fake news

# Combine the datasets
df = pd.concat([df_true, df_fake], ignore_index=True)

# Preprocess the data
df['content'] = df['title'] + ' ' + df['text']
df['content'] = df['content'].apply(preprocess_text)

# Split the data into training and testing sets
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
