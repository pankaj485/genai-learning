from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
import json

load_dotenv()

api_key = getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# client = OpenAI(api_key=api_key, base_url=base_url)
client = OpenAI()


SYSTEM_PROMPT = """
You are an AI Persona Assistant named Alex. 
You are acting on behalf of John who is customer support agent at AWS
John is system administrator and CTO of a startup. 

Rules: 
If the input is not about AWS then don't respond and simply say "sorry, I can't answer that. I'm only AWS expert"

Examples: 
Q: Hey
A: Wassup mate, how can I help you? 
"""


user_query = input("What help you need on AWS? ")


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)


message_respone = response.choices[0].message.content


print(message_respone)
