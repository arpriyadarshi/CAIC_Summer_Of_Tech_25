import os
from dotenv import load_dotenv
import google.generativeai as genai

class AITweetGenerator:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing or not loaded.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash")  # or any model you listed

    def generate_ai_tweet(self, prompt):
        response = self.model.generate_content(prompt)
        tweet = response.text.strip()
        return tweet  # ensure it fits within Twitter's limit
