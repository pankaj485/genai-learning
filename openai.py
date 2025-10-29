from openai import OpenAI
from dotenv import load_dotenv

# will automatically load the API key from .env file
# NOTE: API key should be named exactly as "OPENAI_API_KEY" in .env file
load_dotenv()
client = OpenAI()


response = client.chat.completions.create(
    model="o4-mini", messages=[{"role": "user", "content": "Hey There"}]
)

output_message = response.choices[0].message.content

print(output_message)
