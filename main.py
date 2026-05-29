import os
from datetime import datetime
import os
import uuid
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import gradio as gr
#Used for saving history
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
#Web Search Capability - Third party tool for web search
from langchain_tavily import TavilySearch

load_dotenv()

def get_date():
    """ Get current date """
    return datetime.now().strftime("%Y-%m-%d")

search_tool = TavilySearch()

conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

system_prompt = """
You're an assistance to help answer user's query.
Answer all user queries
if the user asks for date, only then use the get_date tool
use the search_tool for answering questions that require up to date information
"""
llm = ChatOllama(model="gemma4")

agent = create_agent(model=llm, tools=[get_date,search_tool], system_prompt=system_prompt, checkpointer=checkpoint)

def chat(message,history,thread_id):
    config = {"configurable": {"thread_id" : thread_id}}
    response = agent.invoke({"messages": [{"role": "user", "content": message}]},config)
    last_response = response['messages'][-1].content
    return last_response

with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot Agent")
    thread_id = gr.State(value = lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

demo.launch()

#/Users/abhijeetprakash/PycharmProjects/AI_Agent/main.py:21: LangChainDeprecationWarning: The class `TavilySearchResults` was deprecated in LangChain 0.3.25 and will be removed in 1.0. An updated version of the class exists in the `langchain-tavily package and should be used instead. To use it run `pip install -U `langchain-tavily` and import as `from `langchain_tavily import TavilySearch``.
