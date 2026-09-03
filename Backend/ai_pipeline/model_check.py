import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
models = client.models.list()

print("--- AVAILABLE GROQ MODELS ---")
for m in sorted([model.id for model in models.data]):
    print(m)