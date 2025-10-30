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
You're an expert AI Assistant in resolving user queries using chain of thought. 
You work on START, PLAN and OUTPUT steps. 
You need to first PLAN what needs to be done. The PLAN can be multiple steps. 
Once you think enough PLAN has been done, finally you can give OUTPUT.

Rules: 
- strictly follow the given output format
- Only run one step a time. 
- The sequence of steps is START (user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to be displayed to the user.)


Output JSON format: 
{"step": "START" | "PLAN" | "OUTPUT", "content": "string" }

Example: 
START: Hey, Can you solve 2+3*7/10
PLAN:  {"step": "PLAN", "content": "Seems like user is interested in maths problem"}
PLAN:  {"step": "PLAN", "content": "Looking at the problem, we should solve this using BODMAS method"}
PLAN:  {"step": "PLAN", "content": "Yes, BODMAS is corrent method to follow here"}
PLAN:  {"step": "PLAN", "content": "first we must multiply 3*7 which is 21"}
PLAN:  {"step": "PLAN", "content": "Now the new equation is 2 + 21 / 10"}
PLAN:  {"step": "PLAN", "content": "Now the new equation is 2 + 21 / 10"}
PLAN:  {"step": "PLAN", "content": "Now we must perform division. That is 21/10 which is 2.1"}
PLAN:  {"step": "PLAN", "content": "Now the new equation is 2 + 2.1 which is 4.1"}
PLAN:  {"step": "PLAN", "content": "Great, we have finally solved it and result is 4.2"}
PLAN:  {"step": "OUTPUT", "content": "Yes, the result is 4.2"}


"""


message_history = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

user_query = input("Your query is here: ")
message_history.append({"role": "user", "content": user_query})


while True:

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=message_history,
    )

    output_message = response.choices[0].message.content
    message_history.append({"role": "user", "content": output_message})

    parsed_result = json.loads(output_message)

    if parsed_result.get("step") == "START":
        print("Initializing query")
        print(parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "PLAN":
        print(parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "OUTPUT":
        print("Finished processing users query")
        break


""" 
Initializing query
User is asking for the factorial of 5.
The factorial of a number n, denoted as n!, is the product of all positive integers up to n.
In this case, we need to calculate 5! which will be 5 x 4 x 3 x 2 x 1.
Calculating the product step-by-step: 5 x 4 is 20.
Now, multiply 20 by 3, which gives us 60.
Next, multiply 60 by 2, resulting in 120.
Finally, multiply 120 by 1, which remains 120.
Finished processing users query
"""
