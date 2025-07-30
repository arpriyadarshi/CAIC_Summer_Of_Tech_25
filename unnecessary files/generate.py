import requests

url = "http://localhost:5001/generate"  # Your Flask API endpoint

payload = {
    "company": "Nike",
    "tweet_type": "announcement",
    "message": "launching a new AI-powered sneaker",
    "topic": "smart shoes"
}

try:
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Generated Tweet:", result["generated_tweet"])
    else:
        print(f"❌ Failed to generate tweet. Status code: {response.status_code}")
        print("Details:", response.json())

except Exception as e:
    print("❌ Error sending request:", e)
