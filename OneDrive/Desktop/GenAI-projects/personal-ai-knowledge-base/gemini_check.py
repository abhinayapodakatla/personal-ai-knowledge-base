import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key found:", bool(api_key))

client = genai.Client(
    api_key=api_key,
    http_options={
        "timeout": 30000
    }
)

print("Sending request to Gemini...")

try:

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say only: Hello"
    )

    print("Gemini response:")
    print(response.text)

except Exception as error:

    print("Gemini Error:")
    print(type(error).__name__)
    print(error)