import random

class SimpleTweetGenerator:
    def __init__(self):
        self.templates = {
            'announcement': [
                "🚀 Exciting news from {company}! {message}",
                "Big announcement: {company} is {message} 🎉",
                "Hey everyone! {company} has {message} ✨"
            ],
            'question': [
                "What do you think about {topic}? Let us know! 💬",
                "Quick question: How do you feel about {topic}? 🤔",
                "{company} wants to know: What's your take on {topic}? 🗣️"
            ],
            'general': [
                "Check out what {company} is up to! {message} 🌟",
                "{company} update: {message} 💯",
                "From the {company} team: {message} 🔥"
            ]
        }
    
    def generate_tweet(self, company, tweet_type="general", message="Something awesome!", topic="innovation"):
        template_list = self.templates.get(tweet_type, self.templates['general'])
        template = random.choice(template_list)
        tweet = template.format(company=company, message=message, topic=topic)
        return tweet[:280]
