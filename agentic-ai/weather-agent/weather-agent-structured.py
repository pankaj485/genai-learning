from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel, Field
from typing import Optional
import requests
from json import dumps


load_dotenv()


client = OpenAI(api_key=getenv("OPENAI_API_KEY"))


def get_weather(city="bengaluru"):
    format = "%C=%t"
    url = f"https://wttr.in/{city}?format={format}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.text

    return "N/A"


SYSTEM_PROMPT = """
You are a weather-only AI agent.  
Your sole purpose is to provide real-time weather information by calling the function:
    get_weather(city: string) -> takes city name as input and returns back the current temperature of the city or returns "N/A" if city is not valid. 

You work on START, PLAN and OUTPUT steps. 
You need to first PLAN what needs to be done. The PLAN can be multiple steps. 
Once you think enough PLAN has been done, finally you can give OUTPUT.

Rules: 
- strictly follow the given output format
- Only run one step a time. 
- The sequence of steps is START (user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to be displayed to the user.)
- If the user asks anything unrelated to weather, or fails to provide a proper city name, respond with: "I am a weather-only agent. Please provide a valid city name to get the weather."
- Never answer questions outside weather retrieval.
- Never invent city names, and don't ask to confirm the city name. 
- If unsure about any city, use it as it is.  


Output JSON format: 
{"step": "START" | "PLAN" | "OUTPUT" | "TOOL" | "OBSERVE", "content": "string" }

Example: 
START: What is current weather in Bengaluru?

PLAN:  {"step": "PLAN", "content": "Seems like user is interested in getting weather information"}
PLAN:  {"step": "PLAN", "content": "Let's see if we have any available tool in our system"}
PLAN:  {"step": "PLAN", "content": "Great, we have get_weather tool available"}
PLAN:  {"step": "PLAN", "content": "I need to get city name from the user input "}
PLAN:  {"step": "PLAN", "content": "I need tocall get_weather tool with city name"}
PLAN:  {"step": "PLAN", "content": "If tool responds with "N/A" then city is invalid"}

PLAN:  {"step": "TOOL", "tool": "get_weather", "input": "delhi"}
PLAN:  {"step": "OBSERVE", "tool": "get_weather", "output": "Current temperature of 'Dehli' is '26 C' "}
PLAN:  {"step": "OBSERVE", "tool": "get_weather", "output": "We have temperature of 'Delhi' now I will check if there are any other remaining cities."}

PLAN:  {"step": "OUTPUT", "content": "Current temperature of 'Dehli' is '26 C' "}
"""

message_history = [{"role": "system", "content": SYSTEM_PROMPT}]


# validating output format for openai response
class OutputFormat(BaseModel):
    step: str = Field(None, description="ID of the step. Example: 'START','PLAN'")
    content: Optional[str] = Field(None, description="Optional string content")
    tool: Optional[str] = Field(None, description="The ID of the tool to call")
    input: Optional[str] = Field(None, description="Input params for the tool")


available_tools = {"get_weather": get_weather}


while True:
    user_query = input(">> ")
    message_history.append({"role": "user", "content": user_query})

    while True:
        # completions.parse is used so that we get desired format of output
        response = client.chat.completions.parse(
            model="gpt-5-nano",
            response_format=OutputFormat,
            messages=message_history,
        )

        raw_result = response.choices[0].message.content
        message_history.append({"role": "assistant", "content": raw_result})

        parsed_result = response.choices[0].message.parsed

        if parsed_result.step == "START":
            print(f"START: {parsed_result.content}")
            continue

        if parsed_result.step == "TOOL":
            tool_to_call = parsed_result.tool
            tool_input = parsed_result.input
            tool_response = available_tools[tool_to_call](tool_input)

            print(
                f"TOOL: {tool_to_call}, INPUT: {tool_input}, RESPONSE: {tool_response}"
            )

            message_history.append(
                {
                    "role": "developer",
                    "content": dumps(
                        {
                            "step": "OBSERVE",
                            "tool": tool_to_call,
                            "input": tool_input,
                            "output": tool_response,
                        }
                    ),
                }
            )

            continue

        if parsed_result.step == "PLAN":
            print(f"PLAN: {parsed_result.content}")
            continue

        if parsed_result.step == "OUTPUT":
            print(f"OUTPUT: {parsed_result.content}")
            break
