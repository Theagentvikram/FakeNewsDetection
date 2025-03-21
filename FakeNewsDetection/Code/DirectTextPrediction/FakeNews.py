import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib as jb

# Download NLTK stopwords
nltk.download('stopwords')

# Set Streamlit page configuration
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

# Caching the loading of models to optimize performance
@st.cache_resource
def load_model():
    return jb.load("finaldump.joblib")

@st.cache_resource
def load_vectorizer():
    return jb.load("vect.dat")

@st.cache_resource
def load_accuracy():
    return jb.load("acc.dat")

# Load all resources
model = load_model()
vectorizer = load_vectorizer()
acc = load_accuracy()

# Text preprocessing function
def preprocess_text(content):
    stop_words = stopwords.words('english')
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# Prediction function
def predict_fake_news(text):
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Main Streamlit UI
def main():
    st.title("📰 Fake News Detection App")
    st.markdown("Enter the news content below to check whether it's **Real** or **Fake**.")

    text = st.text_area("Enter the news text here:")
    
    if st.button("Predict"):
        if text.strip() == "":
            st.warning("⚠️ Please enter some text before clicking Predict.")
        else:
            prediction = predict_fake_news(text)
            if prediction == 0:
                st.success("✅ The news is **Real**")
            else:
                st.error("❌ The news is **Fake**")

    st.markdown("---")
    st.subheader("📊 Model Accuracy")
    st.write(f"**Training Accuracy:** {acc[0]:.2f}")
    st.write(f"**Testing Accuracy:** {acc[1]:.2f}")

# Run the app
if __name__ == "__main__":
    main()
