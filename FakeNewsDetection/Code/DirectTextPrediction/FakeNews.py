import streamlit as st
import requests

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/bert-base-uncased-fake-news"
headers = {
    "Authorization": f"Bearer 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"
}

# Function to query the model
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection using Hugging Face API")

st.markdown("Enter a news article or sentence to detect if it's **real or fake**.")

user_input = st.text_area("Enter news content:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            output = query({"inputs": user_input})
        
        if isinstance(output, list) and "label" in output[0]:
            label = output[0]["label"]
            score = output[0]["score"]
            
            if label.lower() == "real":
                st.success(f"✅ The news is likely **REAL** (Confidence: {score:.2%})")
            else:
                st.error(f"❌ The news is likely **FAKE** (Confidence: {score:.2%})")
        else:
            st.error("⚠️ Error from model or rate limit exceeded.")
            st.json(output)
