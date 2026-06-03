import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from supervisor.prompts import build_system_prompt


def build_agent_node(tools: list):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    ).bind_tools(tools)

    def agent_node(state: dict) -> dict:
        system_prompt = build_system_prompt(state)
        messages      = [SystemMessage(content=system_prompt)] + state["messages"]
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