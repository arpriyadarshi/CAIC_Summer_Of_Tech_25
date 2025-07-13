from flask import Flask, request, jsonify
from tweet_generator import SimpleTweetGenerator
from bonus_ai_generator import AITweetGenerator  
import requests
# Optional

app = Flask(__name__)
template_generator = SimpleTweetGenerator()
ai_generator = AITweetGenerator()

@app.route('/')
def home():
    return "✅ Tweet Generator API is running. Use POST /generate"


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        company = data.get('company', 'Our Company')
        tweet_type = data.get('tweet_type', 'general')
        message = data.get('message', 'Something awesome!')
        topic = data.get('topic', 'innovation')

        tweet = template_generator.generate_tweet(company, tweet_type, message, topic)

        return jsonify({
            'generated_tweet': tweet,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/generate_ai', methods=['POST'])
def generate_ai():
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'The future of AI is')
        ai_tweet = ai_generator.generate_ai_tweet(prompt)

        return jsonify({
            'generated_tweet': ai_tweet,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/generate_and_predict', methods=['POST'])
def generate_and_predict():
    try:
        data = request.get_json()

        # Extract inputs for generation
        company = data.get('company', 'Our Company')
        tweet_type = data.get('tweet_type', 'general')
        message = data.get('message', 'Something awesome!')
        topic = data.get('topic', 'innovation')
        date = data.get('date', '2025-06-17 09:30:00')
        media = data.get('media', 'no_media')

        # Step 1: Generate tweet
        generated_tweet = template_generator.generate_tweet(company, tweet_type, message, topic)

        # Step 2: Build payload for predictor
        predictor_input = {
            "content": generated_tweet,
            "date": date,
            "media": media,
            "inferred company": company
        }

        # Step 3: Send to Likes Predictor API (port 5000)
        response = requests.post("http://127.0.0.1:5000/predict", json=predictor_input)

        if response.status_code == 200:
            prediction = response.json()
            return jsonify({
                "generated_tweet": generated_tweet,
                "predicted_likes": prediction["predicted_likes"],
                "log_likes": prediction["log_likes"],
                "success": True
            })
        else:
            return jsonify({
                "error": "Prediction API returned an error.",
                "details": response.text,
                "success": False
            }), 500

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/generate_ai_and_predict', methods=['POST'])
def generate_ai_and_predict():
    try:
        data = request.get_json()

        # Check for required fields explicitly
        required_fields = ["company", "message", "tweet_type", "topic", "date", "media"]
        missing = [field for field in required_fields if field not in data or not data[field]]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing)}",
                "success": False
            }), 400

        company = data["company"]
        message = data["message"]
        tweet_type = data["tweet_type"]
        topic = data["topic"]
        date = data["date"]
        media = data["media"]

        # Build a clear and natural prompt using the input values
        prompt = (
            f"Compose a meaningful, complete tweet (max 280 characters) for {company} "
            f"announcing: {message}. The tweet should be about {topic}, engaging, "
            f"and suitable for Twitter. Avoid cutting off mid-sentence."
        )

        # Step 1: Generate tweet from AI
        generated_tweet = ai_generator.generate_ai_tweet(prompt)

        # Step 2: Prepare data for prediction
        predictor_input = {
            "content": generated_tweet,
            "date": date,
            "media": media,
            "inferred company": company
        }

        # Step 3: Send to Likes Predictor API
        response = requests.post("http://127.0.0.1:5000/predict", json=predictor_input)

        if response.status_code == 200:
            prediction = response.json()
            return jsonify({
                "generated_tweet": generated_tweet,
                "predicted_likes": prediction["predicted_likes"],
                "log_likes": prediction["log_likes"],
                "success": True
            })
        else:
            return jsonify({
                "error": "Prediction API returned an error.",
                "details": response.text,
                "success": False
            }), 500

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Tweet Generator API is running!'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
