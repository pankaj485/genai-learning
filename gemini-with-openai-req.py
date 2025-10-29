from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()

api_key = getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = OpenAI(api_key=api_key, base_url=base_url)

response = client.chat.completions.create(
    model="gemini-2.5-flash", messages=[{"role": "user", "content": "Hey There"}]
)

message_response = response.choices[0].message

print(message_response)
