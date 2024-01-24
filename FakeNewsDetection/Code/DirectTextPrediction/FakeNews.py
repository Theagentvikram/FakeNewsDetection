import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib as jb

# Download NLTK stopwords
nltk.download('stopwords')

# Load pre-trained models
model = jb.load("finaldump.joblib")
vectorizer = jb.load("vect.dat")
st.set_page_config(page_title="Fake News Detection", page_icon="IMG.png")
# Preprocess text
def preprocess_text(content):
    stop_words = stopwords.words('english')
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Predict fake news
def predict_fake_news(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Streamlit app code
def main():
    st.title("Fake News Detection")

    text = st.text_input("Enter the news text:")
    if st.button("Predict"):
        if text:
            prediction = predict_fake_news(text)
            if prediction == 0:
                st.write("The news is Real")
            else:
                st.write("The news is Fake")
    acc=jb.load("acc.dat")
    st.write("Training Accuracy:", acc[0])
    st.write("Testing Accuracy:", acc[1])

if __name__ == "__main__":
    main()

