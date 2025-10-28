import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

print(llm.invoke("Hello, world!"))