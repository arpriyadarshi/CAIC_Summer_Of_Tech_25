import requests

url = "http://127.0.0.1:5001/generate_ai"  # URL of your Flask app

data = {
    "prompt": "Nike just launched its new AI-powered sneakers"  # Change this prompt as needed
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("✅ Generated Tweet:", result['generated_tweet'])
else:
    print("❌ Failed to generate tweet. Status code:", response.status_code)
    print("Details:", response.text)
