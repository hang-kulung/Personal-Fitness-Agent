from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from prompts import build_system_prompt
import os
from dotenv import load_dotenv

load_dotenv()


def build_agent_node(tools: list):
    """
    Returns a node function with the LLM and tools baked in.
    We use a factory so graph.py can pass the tool list cleanly.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(tools)   # bind_tools is the LangChain equivalent of listing tools on Agent

    def agent_node(state: dict) -> dict:
        system_prompt = build_system_prompt(state)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response  = llm.invoke(messages)
        return {"messages": [response]}  # appended to state["messages"] automatically

    return agent_node