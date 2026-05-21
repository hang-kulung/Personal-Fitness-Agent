import asyncio
# import os
# from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import build_graph

# load_dotenv()

USER_ID   = "user_001"
THREAD_ID = f"{USER_ID}_session"   # one persistent thread per user
DB_PATH   = "workout_agent.db"


async def main():
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph  = build_graph(checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}

        print("Workout Trainer Agent  |  type 'exit' to quit\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("Agent: ", end="", flush=True)

            # Pass user_id + thread_id into initial state once
            # After first turn the checkpointer carries state forward
            async for event in graph.astream(
                {
                    "messages":  [HumanMessage(content=user_input)],
                    "user_id":   USER_ID,
                    "thread_id": THREAD_ID,
                },
                config=config,
                stream_mode="values",
            ):
                # Print the last AI message when it appears
                messages = event.get("messages", [])
                if messages and hasattr(messages[-1], "content"):
                    last = messages[-1]
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        continue    # skip intermediate tool-call messages
                    if last.content and last.__class__.__name__ == "AIMessage":
                        print(last.content)


if __name__ == "__main__":
    asyncio.run(main())