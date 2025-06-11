from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import re
import ftfy
import emoji

app = Flask(__name__)

# Load model and components
components = joblib.load("xgb_pipeline.pkl")
model = components['model']
scaler = components['scaler']
tfidf = components['tfidf']
le = components['label_encoder']
company_means = components['company_means']
company_likes_map = components['company_likes_map']

bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Loaded but not used here

def preprocess(data):
    df = pd.DataFrame([data])
    df['content'] = df['content'].astype(str).str.strip().str.lower()
    df['media'] = df.get('media', 'no_media')
    df['new_content'] = df['content'].apply(ftfy.fix_text).apply(emoji.demojize)
    df['new_content1'] = df['content'].apply(ftfy.fix_text)
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')

    # Time features
    df['hour'] = df['datetime'].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df['datetime'].dt.weekday.fillna(0).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour'].between(0, 6).astype(int)

    # Text features
    df['char_len'] = df['new_content'].str.len()
    df['word_count'] = df['new_content'].str.split().apply(len)
    df['avg_word_length'] = df['new_content'].apply(lambda x: sum(len(w) for w in x.split()) / max(1, len(x.split())))
    df['has_hashtag'] = df['new_content'].str.contains('#').astype(int)
    df['has_mention'] = df['new_content'].str.contains('<mention>').astype(int)
    df['has_link'] = df['new_content'].str.contains('<hyperlink>').astype(int)

    # Sentiment
    df['polarity'] = df['new_content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['has_positive_sentiment'] = (df['polarity'] > 0).astype(int)

    # Emoji features
    def count_emojis(text):
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        return len(emoji_pattern.findall(text))
    df['emoji_count'] = df['new_content1'].map(count_emojis)
    df['has_emoji'] = (df['emoji_count'] > 0).astype(int)

    # Post length
    df['is_long_post'] = (df['char_len'] > 200).astype(int)

    # Media flags
    def extract_media_flags(media_str):
        if not isinstance(media_str, str):
            return pd.Series([0, 0, 0])
        return pd.Series([
            int("Photo(" in media_str),
            int("Video(" in media_str),
            int("AnimatedGif(" in media_str or "Gif(" in media_str)
        ])
    df[['has_photo', 'has_video', 'has_gif']] = df['media'].apply(extract_media_flags)

    # Company-level features
    company = df['inferred company'].iloc[0].strip().lower()
    df['company_avg_log_likes'] = company_means.get(company, company_means.mean())
    df['company_popularity'] = company_likes_map.get(company, company_means.median())
    df['company_encoded'] = le.transform([company])[0] if company in le.classes_ else 0

    # Feature groups
    num_cols = ['hour', 'char_len', 'word_count', 'polarity', 'day_of_week',
                'company_avg_log_likes', 'emoji_count', 'avg_word_length']
    binary_cols = ['is_weekend', 'has_hashtag', 'has_mention', 'has_link',
                   'has_photo', 'has_video', 'has_gif', 'has_emoji',
                   'is_long_post', 'has_positive_sentiment', 'is_night']

    # Scale numeric columns
    df[num_cols] = scaler.transform(df[num_cols])

    # TF-IDF
    tfidf_matrix = tfidf.transform(df['new_content'].fillna('')).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf.get_feature_names_out())

    # Final feature matrix
    X = pd.concat([df[num_cols + binary_cols].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return X

@app.route('/')
def home():
    return "📍 Likes Prediction API is up! Use POST /predict"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X = preprocess(data)
        log_likes = model.predict(X.values)[0]
        predicted_likes = int(np.expm1(log_likes))
        return jsonify({
            "log_likes": float(round(log_likes, 3)),
            "predicted_likes": int(predicted_likes)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("✅ Flask API starting at http://127.0.0.1:5000 ...")
    app.run(debug=True)
