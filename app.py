import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import ftfy
import emoji
from datetime import datetime
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from bonus_ai_generator import AITweetGenerator

# ───────────────────────────── Load Env and Configure ─────────────────────────────
load_dotenv()

st.set_page_config(page_title="AI Tweet Generator + Likes Predictor", layout="centered")
st.title("📢 AI Tweet Generator + Likes Predictor")

# ───────────────────────────── Load Models ─────────────────────────────
@st.cache_resource
def load_components():
    components = joblib.load("xgb_pipeline.pkl")
    model = components['model']
    scaler = components['scaler']
    tfidf = components['tfidf']
    le = components['label_encoder']
    company_means = components['company_means']
    company_likes_map = components['company_likes_map']
    return model, scaler, tfidf, le, company_means, company_likes_map

model, scaler, tfidf, le, company_means, company_likes_map = load_components()

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = AITweetGenerator()

# ───────────────────────────── Helpers ─────────────────────────────

def preprocess(data):
    df = pd.DataFrame([data])
    df['content'] = df['content'].astype(str).str.strip().str.lower()
    df['media'] = df.get('media', 'no_media')
    df['new_content'] = df['content'].apply(ftfy.fix_text).apply(emoji.demojize)
    df['new_content1'] = df['content'].apply(ftfy.fix_text)
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

    df['hour'] = df['datetime'].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df['datetime'].dt.weekday.fillna(0).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(0, 6).astype(int)

    df['char_len'] = df['new_content'].str.len()
    df['word_count'] = df['new_content'].str.split().apply(len)
    df['avg_word_length'] = df['new_content'].apply(lambda x: sum(len(w) for w in x.split()) / max(1, len(x.split())))
    df['has_hashtag'] = df['new_content'].str.contains('#').astype(int)
    df['has_mention'] = df['new_content'].str.contains('<mention>').astype(int)
    df['has_link'] = df['new_content'].str.contains('<hyperlink>').astype(int)
    df['polarity'] = df['new_content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['has_positive_sentiment'] = (df['polarity'] > 0).astype(int)

    def count_emojis(text):
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        return len(emoji_pattern.findall(text))
    df['emoji_count'] = df['new_content1'].map(count_emojis)
    df['has_emoji'] = (df['emoji_count'] > 0).astype(int)
    df['is_long_post'] = (df['char_len'] > 200).astype(int)

    def extract_media_flags(media_str):
        if not isinstance(media_str, str):
            return pd.Series([0, 0, 0])
        return pd.Series([
            int("Photo(" in media_str),
            int("Video(" in media_str),
            int("AnimatedGif(" in media_str or "Gif(" in media_str)
        ])
    df[['has_photo', 'has_video', 'has_gif']] = df['media'].apply(extract_media_flags)

    company = df['inferred company'].iloc[0].strip().lower()
    df['company_avg_log_likes'] = company_means.get(company, company_means.mean())
    df['company_popularity'] = company_likes_map.get(company, company_means.median())
    df['company_encoded'] = le.transform([company])[0] if company in le.classes_ else 0

    num_cols = ['hour', 'char_len', 'word_count', 'polarity', 'day_of_week',
                'company_avg_log_likes', 'emoji_count', 'avg_word_length']
    binary_cols = ['is_weekend', 'has_hashtag', 'has_mention', 'has_link',
                   'has_photo', 'has_video', 'has_gif', 'has_emoji',
                   'is_long_post', 'has_positive_sentiment', 'is_night']

    df[num_cols] = scaler.transform(df[num_cols])
    tfidf_matrix = tfidf.transform(df['new_content'].fillna('')).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf.get_feature_names_out())

    X = pd.concat([df[num_cols + binary_cols].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return X

def predict_likes(content, inferred_company, date, media):
    data = {
        "content": content,
        "inferred company": inferred_company,
        "date": date,
        "media": media
    }
    X = preprocess(data)
    log_likes = model.predict(X.values)[0]
    return int(np.expm1(log_likes))

# ───────────────────────────── Streamlit UI ─────────────────────────────

st.subheader("🧠 Generate AI Tweet + Predict Likes")

if st.button("🔄 Reset Fields", key="reset_generator_fields"):
    for key in ["company", "tweet_type", "message", "topic"]:
        st.session_state[key] = ""

company = st.text_input("Company", key="company")
tweet_type = st.selectbox("Tweet Type", ["announcement", "question", "general"], key="tweet_type")
message = st.text_area("Message", key="message")
topic = st.text_input("Topic", key="topic")



if st.button("✨ Generate Tweet & Predict Likes"):
    generated = generator.generate_ai_tweet(f"Write a highly engaging tweet for {company} about {topic}. "
    f"The tweet should be in the tone of a {tweet_type}. "
    f"Include this message: '{message}'. "
    f"Only return the tweet. Do not explain or list multiple options. Do not fact-check. use hashtags meaningfully. may use emojis if appropriate. within 280 characters.")
    predicted = predict_likes(generated, company, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "no_media")
    st.success("✅ Generated Tweet:")
    st.text_area("Generated Tweet", generated, height=150)
    st.info(f"📊 Predicted Likes: {predicted}")

st.markdown("---")

st.subheader("🧲 Predict Likes from Manual Tweet")

if st.button("🔄 Reset Fields", key="reset_manual_fields"):
    for key in ["manual_content", "manual_company", "manual_date", "manual_media"]:
        st.session_state[key] = ""

manual_content = st.text_area("Tweet Content", key="manual_content")
manual_company = st.text_input("Inferred Company", key="manual_company")
manual_date = st.text_input("Tweet Date (YYYY-MM-DD HH:MM:SS)", key="manual_date")
manual_media = st.text_input("Media Type (e.g., Photo, Video)", key="manual_media")

if st.button("📈 Predict Likes Only"):
    predicted = predict_likes(manual_content, manual_company, manual_date, manual_media)
    st.info(f"📊 Predicted Likes: {predicted}")
