import requests

payload = {
    "prompt": "OpenAI is about to change the world again!",
    "date": "2025-06-17 10:00:00",
    "media": "Photo(openai_event.jpg)",
    "company": "OpenAI"
}

res = requests.post("http://127.0.0.1:5001/generate_ai_and_predict", json=payload)

if res.status_code == 200:
    print("🤖 Tweet:", res.json()['generated_tweet'])
    print("❤️ Predicted Likes:", res.json()['predicted_likes'])
else:
    print("❌ Error:", res.text)
