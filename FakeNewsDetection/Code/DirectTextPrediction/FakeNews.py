import streamlit as st
import requests

# ✅ Hugging Face Inference API details
API_URL = "https://api-inference.huggingface.co/models/jy46604790/Fake-News-Bert-Detect"
HF_TOKEN = "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"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ✅ Function to query Hugging Face model safely
def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as err:
        st.error(f"⚠️ Request failed: {err}")
        return None

# ✅ Streamlit UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection using Hugging Face")

st.markdown("Enter a news article or statement and the model will predict whether it's **Fake** or **Real**.")

user_input = st.text_area("📝 News Content", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        with st.spinner("Analyzing..."):
            result = query({"inputs": user_input})

        if result and isinstance(result, list) and isinstance(result[0], list):
            predictions = result[0]
            top = max(predictions, key=lambda x: x["score"])
            label = top["label"]
            score = top["score"]

            if label == "LABEL_0":
                st.error(f"❌ This news is likely **FAKE**\nConfidence: {score:.2%}")
            elif label == "LABEL_1":
                st.success(f"✅ This news is likely **REAL**\nConfidence: {score:.2%}")
            else:
                st.info(f"ℹ️ Result: {label} ({score:.2%})")
        else:
            st.error("⚠️ Unexpected model response.")
            st.json(result)
