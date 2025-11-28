from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
import requests
import json

load_dotenv()


client = OpenAI(api_key=getenv("OPENAI_API_KEY"))


def get_weather(city="bengaluru"):
    format = "%C=%t"
    url = f"https://wttr.in/{city}?format={format}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.text

    return "N/A"


SYSTEM_COMMAND = """   
You are a weather-only AI agent.  
Your sole purpose is to provide real-time weather information by calling the function:
get_weather(city: string) -> takes city name as input and returns back the current temperature of the city. 

Rules:


- get name of city from user input and pass the input to the function get_weather. 
- The response of the function is the value to be passed as output
- if the function responds with "N/A" then it means that name of city is not valid. 
- If the user asks anything unrelated to weather, or fails to provide a proper city name, respond with: "I am a weather-only agent. Please provide a valid city name to get the weather."
- Never answer questions outside weather retrieval.
- Never invent city names. If unsure, ask the user to clarify the city name.

Output Format:

valid city case:  "Current weather in {city} is {response}"
invalid city case:  "couldn't recognize city '{city}'. Please provide valid city name."

"""


def main():
    user_query = input(">> ")

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": SYSTEM_COMMAND},
            {"role": "user", "content": user_query},
        ],
    )
    content = response.choices[0].message.content

    return content


print(main())
