from openai import OpenAI
from dotenv import load_dotenv
from os import getenv
import json

load_dotenv()

api_key = getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = OpenAI(api_key=api_key, base_url=base_url)


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


response = client.chat.completions.create(
    model="gemini-2.5-flash",
    response_format={"type": "json_schema"},
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hey, write a code to add n numbers in js"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "START",
                    "content": "You want a javascript code to add 'n' numbers",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "content": "The user wants a JavaScript function to add an 'n' number of arguments.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "content": "I should define a function that can accept a variable number of arguments.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "content": "I can use the rest parameter syntax (...) to gather all arguments into an array.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "content": "Inside the function, I will iterate over the array of numbers and sum them up.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {"step": "PLAN", "content": "I will initialize a sum variable to 0."}
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "content": "I will use a for...of loop to iterate through the numbers array and add each number to the sum.",
                }
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "step": "PLAN",
                    "content": "Finally, I will return the calculated sum.",
                }
            ),
        },
    ],
)

output_message = response.choices[0].message.content

print(output_message)

""" 
{
    "step": "OUTPUT",
    "content": "```javascript\nfunction addNumbers(...numbers) {\n  let sum = 0;\n  for (const number of numbers) {\n    sum += number;\n  }\n  return sum;\n}\n\n// Example usage:\nconsole.log(addNumbers(1, 2, 3));        // Output: 6\nconsole.log(addNumbers(10, 20, 30, 40)); // Output: 100\nconsole.log(addNumbers(5));             // Output: 5\nconsole.log(addNumbers());              // Output: 0\nconsole.log(addNumbers(1.5, 2.5, 3));   // Output: 7\n```"
}
"""
