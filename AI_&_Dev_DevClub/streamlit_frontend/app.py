import streamlit as st
import requests

st.set_page_config(page_title="AI Tweet Generator + Likes Predictor", layout="centered")

st.title("📢 AI Tweet Generator + Likes Predictor")

# -------------------- 1. AI Generation + Likes Prediction --------------------
st.subheader("🧠 Generate AI Tweet + Predict Likes")

company = st.text_input("Company", "OpenAI")
tweet_type = st.selectbox("Tweet Type", ["announcement", "question", "general"])
message = st.text_area("Message", "just launched GPT-5!")
topic = st.text_input("Topic", "AI")
date = st.text_input("Date (YYYY-MM-DD HH:MM:SS)", "2025-06-17 09:30:00")
media = st.text_input("Media Type (e.g., Photo, Video)", "Photo")

if st.button("✨ Generate AI Tweet and Predict Likes"):
    prompt = f"{company} {tweet_type} {message} about {topic}"
    payload = {
        
        "company": company,
        "message": message,
        "tweet_type": tweet_type,
        "topic": topic,
        "date": date,
        "media": media
    }

    try:
        response = requests.post("http://127.0.0.1:5001/generate_ai_and_predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Generated Tweet:")
            st.text_area("Generated Tweet Output", value=result['generated_tweet'], height=150)
            st.info(f"📊 Predicted Likes: {result['predicted_likes']}")

        else:
            st.error("❌ Error from API:")
            st.json(response.json())
    except Exception as e:
        st.error(f"⚠️ Exception occurred: {e}")

st.markdown("---")

# -------------------- 2. Likes Prediction Only --------------------
st.subheader("🧮 Predict Likes from Manual Tweet")

content = st.text_area("Tweet Content")
inferred_company = st.text_input("Inferred Company", "OpenAI")
date2 = st.text_input("Tweet Date (YYYY-MM-DD HH:MM:SS)", "2025-06-17 09:30:00")
media2 = st.text_input("Media", "Photo(openai_launch.jpg)")

if st.button("📈 Predict Likes Only"):
    payload = {
        "content": content,
        "inferred company": inferred_company,
        "date": date2,
        "media": media2
    }

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"📊 Predicted Likes: {result['predicted_likes']}")
        else:
            st.error("❌ Error from Likes Predictor API:")
            st.json(response.json())
    except Exception as e:
        st.error(f"⚠️ Exception occurred: {e}")
