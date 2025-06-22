import requests

payload = {
    "company": "OpenAI",
    "tweet_type": "announcement",
    "message": "just launched GPT-5!",
    "topic": "AI",
    "date": "2025-06-17 09:30:00",
    "media": "Photo(openai_launch.jpg)"
}

res = requests.post("http://127.0.0.1:5001/generate_and_predict", json=payload)

if res.status_code == 200:
    print("✅ Tweet:", res.json()['generated_tweet'])
    print("📈 Predicted Likes:", res.json()['predicted_likes'])
else:
    print("❌ Error:", res.text)
