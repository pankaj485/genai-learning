from openai import OpenAI
from dotenv import load_dotenv
from os import getenv

import json

load_dotenv()


client = OpenAI(
    api_key=getenv("OPENAI_API_KEY"),
)


respone = client.responses.create(
    model="chatgpt-4o-latest",
    input="Hello, can you help me with prompt serialization in alpaca format with example?",
)

output = respone.output[0].content[0].text

print(output)
