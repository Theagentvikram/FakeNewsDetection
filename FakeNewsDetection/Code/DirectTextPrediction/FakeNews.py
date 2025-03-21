import streamlit as st
import requests

# ✅ Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-mini-finetuned-fake-news"
headers = {
    "Authorization": f"Bearer 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"
}

# 🔁 Inference function
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 🚀 Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection (Hugging Face API)")
st.markdown("Enter any news text below. The model will predict whether it’s **Fake** or **Real**.")

user_input = st.text_area("📝 News Content", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing with BERT model..."):
            output = query({"inputs": user_input})

        if isinstance(output, list) and "label" in output[0]:
            label = output[0]["label"]
            score = output[0]["score"]

            if label.upper() == "FAKE":
                st.error(f"❌ This news is likely **FAKE**\nConfidence: {score:.2%}")
            elif label.upper() == "REAL":
                st.success(f"✅ This news is likely **REAL**\nConfidence: {score:.2%}")
            else:
                st.info(f"ℹ️ Prediction: {label} ({score:.2%})")
        else:
            st.error("⚠️ Error from model or rate limit exceeded.")
            st.json(output)
