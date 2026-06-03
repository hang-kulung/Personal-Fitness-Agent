# supervisor/main.py
import asyncio
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from supervisor.graph import build_graph   # ← after sys.path is set

USER_ID   = "user_001"
THREAD_ID = f"{USER_ID}_supervisor"
DB_PATH   = os.path.join(PROJECT_ROOT, "data", "supervisor.db")


async def main():
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph  = build_graph(checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        print("Personal Fitness Agent  |  type 'exit' to quit\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("Agent: ", end="", flush=True)

            final_response = None

            async for event in graph.astream(
                {
                    "messages":  [HumanMessage(content=user_input)],
                    "user_id":   USER_ID,
                    "thread_id": THREAD_ID,
                },
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in event.items():
                    if not node_output:
                        continue
                    messages = node_output.get("messages", [])
                    if messages:
                        last = messages[-1]
                        if last.__class__.__name__ == "AIMessage":
                            if not (hasattr(last, "tool_calls") and last.tool_calls):
                                final_response = last.content

            if final_response:
                if isinstance(final_response, str):
                    print(final_response)
                elif isinstance(final_response, list):
                    for block in final_response:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block["text"])


if __name__ == "__main__":
    asyncio.run(main())