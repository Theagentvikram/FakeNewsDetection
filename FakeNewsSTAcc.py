import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

nltk.download('stopwords')

def preprocess_text(content):
    ps = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_fake_news(news_content):
    preprocessed_content = preprocess_text(news_content)
    vectorized_content = vectorizer.transform([preprocessed_content])
    prediction = model.predict(vectorized_content)
    return prediction[0]

# Load the dataset
true_df = pd.read_csv(r"C:\Users\abhic\Downloads\True.csv")
fake_df = pd.read_csv(r"C:\Users\abhic\Downloads\Fake.csv")

# Merge the datasets
df = pd.concat([true_df, fake_df])

# Preprocess the content
df['content'] = df['title'] + ' ' + df['text'] + ' ' + df['date']
df['content'] = df['content'].apply(preprocess_text)

# Prepare the data for training
x = df['content'].values
y = df['label'].values

# Vectorize the textual data
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title("Fake News Detection")
st.write("Enter the news content below:")

# User input
news_content = st.text_area("News Content", "")

# Perform prediction when the user clicks the "Predict" button
if st.button("Predict"):
    if news_content:
        prediction = predict_fake_news(news_content)
        if prediction == 0:
            st.write("The news is Real")
        else:
            st.write("The news is Fake")
    else:
        st.warning("Please enter some news content.")

# Display accuracy
st.write(f"Model Accuracy: {accuracy:.2%}")
