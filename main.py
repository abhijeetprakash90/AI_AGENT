from datetime import datetime
from google.genai._interactions.types import content
from langchain.agents import create_agent
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os


#Env Variables Updates
load_dotenv()
#google_api_key = os.getenv("GOOGLE_API_KEY")


def get_date():
    """ Get current date """
    return datetime.now().strftime("%Y-%m-%d")


system_prompt = """
You're an assistance to help answer user's query.
if the user asks date, you can use the get_date tool
"""
llm = ChatOllama(model="qwen2.5:3b")

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)

user_query = input("Enter your query:")
response = agent.invoke({"messages": [{"role" : "user", "content" : user_query}]})

print(response['messages'][1].content)

#Gemini response print method
#print(response['messages'][-1].content[0]['text'])
