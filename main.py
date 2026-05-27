from datetime import datetime
from google.genai._interactions.types import content
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

from pyexpat import model

#Env Variables Updates
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

def get_date():
    """ Get current date """
    return datetime.now().strftime("%Y-%m-%d")

system_prompt = """
You're an assistance to help answer user's query.
if the user asks date, you can use the get_date tool
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)

user_query = input("What would you like to ask?")
response = agent.invoke({"messages": [{"role" : "user", "content" : user_query}]})

print(response['messages'][-1].content[0]['text'])
