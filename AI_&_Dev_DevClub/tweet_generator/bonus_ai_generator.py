from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class AITweetGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_ai_tweet(self, prompt, max_length=60):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()[:280]
