import streamlit as st
import requests

st.title("📢 AI Tweet Generator + Likes Predictor")

st.subheader("Tweet Generation + Likes Prediction")
company = st.text_input("Company", "OpenAI")
tweet_type = st.selectbox("Tweet Type", ["announcement", "question", "general"])
message = st.text_area("Message", "just launched GPT-5!")
topic = st.text_input("Topic", "AI")
date = st.text_input("Date (YYYY-MM-DD HH:MM:SS)", "2025-06-17 09:30:00")
media = st.text_input("Media", "Photo(openai_launch.jpg)")

if st.button("Generate Tweet and Predict Likes"):
    payload = {
        "company": company,
        "tweet_type": tweet_type,
        "message": message,
        "topic": topic,
        "date": date,
        "media": media
    }
    response = requests.post("http://127.0.0.1:5001/generate_and_predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        st.success("✅ Generated Tweet:")
        st.write(result['generated_tweet'])
        st.info(f"📊 Predicted Likes: {result['predicted_likes']}")
    else:
        st.error("❌ Error:")
        st.json(response.json())

st.markdown("---")

### 🧮 Likes Prediction Only
st.subheader("Likes Prediction from Manual Tweet")

content = st.text_area("Tweet Content")
inferred_company = st.text_input("Inferred Company", "OpenAI")
date2 = st.text_input("Tweet Date (YYYY-MM-DD HH:MM:SS AM/PM)", "2025-06-17 09:30:00")
media2 = st.text_input("Media Info", "Photo(openai_launch.jpg)")

if st.button("Predict Likes Only"):
    payload = {
        "content": content,
        "inferred company": inferred_company,
        "date": date2,
        "media": media2
    }
    response = requests.post("http://127.0.0.1:5000/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        st.success(f"📊 Predicted Likes: {result['predicted_likes']}")
    else:
        st.error("❌ Error:")
        st.json(response.json())
