import requests

# URL of your local Flask API
url = "http://127.0.0.1:5000/predict"

# Example input data
data = {
    "content": "Trump is on a Nixonian collision course with the FBI <hyperlink> (via <mention>) <hyperlink>",
    "date": "01-02-2018  04:05:06 AM",
    "media": "[Photo(previewUrl='https://pbs.twimg.com/media/DU6_RI-WsAAdbDI?format=jpg&name=small', fullUrl='https://pbs.twimg.com/media/DU6_RI-WsAAdbDI?format=jpg&name=large')]",
    "inferred company": "cnn"
}

# Send POST request to the API
response = requests.post(url, json=data)

# Print the prediction result
if response.status_code == 200:
    print("✅ Predicted Likes:", response.json()["predicted_likes"])
else:
    print("❌ Failed to get prediction. Status code:", response.status_code)
    print("Details:", response.text)
