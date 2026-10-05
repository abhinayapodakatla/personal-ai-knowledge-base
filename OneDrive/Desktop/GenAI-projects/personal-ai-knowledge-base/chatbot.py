import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API key not found!")
    exit()

client = genai.Client(api_key=api_key)

print("🤖 Gemini AI is ready!")
print("Type 'exit' to stop.")

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye! 👋")
        break

    try:
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=question
        )

        print("\n🤖 Gemini:", interaction.output_text)

    except Exception as error:
        print("\n❌ Error:")
        print(error)