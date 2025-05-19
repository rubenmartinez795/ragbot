import os

import google.generativeai as genai
from IPython.core.display import display_markdown

# Set up Google AI Studio API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Set up GenAI model
model = genai.GenerativeModel("gemini-1.5-flash")

# Start chat
while True:
    prompt = input(">>> ")
    if prompt.lower() == "exit":
        print("Session closed!")
        exit(0)
    response = model.generate_content(prompt)
    display_markdown(response.text)
