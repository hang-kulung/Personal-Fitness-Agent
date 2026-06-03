from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from workout_agent.prompts import build_system_prompt

import os
import time
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
        for attempt in range(3):
            try:
                response = llm.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)   # 30s, 60s
                    print(f"\n[Rate limited — waiting {wait}s...]")
                    time.sleep(wait)
                else:
                    raise
    return agent_node