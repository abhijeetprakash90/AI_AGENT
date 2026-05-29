from datetime import datetime
import uuid
#from anyio.lowlevel import checkpoint
#from google.genai._interactions.types import content
from langchain.agents import create_agent
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import gradio as gr
#Used for saving history
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


#Env Variables Updates
load_dotenv()
#google_api_key = os.getenv("GOOGLE_API_KEY")


def get_date():
    """ Get current date """
    return datetime.now().strftime("%Y-%m-%d")

conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

system_prompt = """
You're an assistance to help answer user's query.
Answer all user queries
if the user asks date, you can use the get_date tool
"""
llm = ChatOllama(model="qwen2.5:3b")

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt, checkpointer=checkpoint)

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