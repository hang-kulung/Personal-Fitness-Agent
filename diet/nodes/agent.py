import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from prompts import build_system_prompt
from dotenv import load_dotenv

load_dotenv()

def build_agent_node(tools: list):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(tools)

    def agent_node(state: dict) -> dict:
        system_prompt = build_system_prompt(state)
        messages      = [SystemMessage(content=system_prompt)] + state["messages"]
        response      = llm.invoke(messages)
        return {"messages": [response]}

    return agent_node