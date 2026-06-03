import asyncio
import os
import sys
from langchain_core.tools import tool

# ── make subagent packages importable from supervisor/ ───────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
WORKOUT_PATH = os.path.join(PROJECT_ROOT, "workout_agent")
DIET_PATH    = os.path.join(PROJECT_ROOT, "diet_agent")

for path in (PROJECT_ROOT, WORKOUT_PATH, DIET_PATH):
    if path not in sys.path:
        sys.path.insert(0, path)

# ── import each agent's graph builder ────────────────────────────────────────
# We import lazily inside the tool functions to avoid circular import issues
# and to defer checkpointer creation until the tools are actually called.

WORKOUT_THREAD = "supervisor_workout_sub"
DIET_THREAD    = "supervisor_diet_sub"


def _run_subagent(graph, message: str, thread_id: str) -> str:
    """
    Synchronously invoke a compiled subagent graph with a single message.
    Returns the final text response.
    """
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": thread_id}}

    # subagent graphs are sync-invoked from the supervisor
    result = graph.invoke(
        {
            "messages":  [HumanMessage(content=message)],
            "user_id":   "user_001",
            "thread_id": thread_id,
        },
        config=config,
    )

    # extract final AI text from result messages
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    return " ".join(
                        b["text"] for b in msg.content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
    return "No response from subagent."


# ── lazy graph cache — built once, reused across tool calls ──────────────────
_workout_graph = None
_diet_graph    = None


def _get_workout_graph():
    global _workout_graph
    if _workout_graph is None:
        from langgraph.checkpoint.memory import MemorySaver  # ← correct
        import workout_agent.graph as wg
        _workout_graph = wg.build_graph(MemorySaver())
    return _workout_graph


def _get_diet_graph():
    global _diet_graph
    if _diet_graph is None:
        from langgraph.checkpoint.memory import MemorySaver  # ← correct
        import diet_agent.graph as dg
        _diet_graph = dg.build_graph(MemorySaver())
    return _diet_graph

# ── the two tools the supervisor LLM can call ─────────────────────────────────

@tool
def ask_workout_agent(message: str) -> str:
    """
    Send a workout or exercise related question to the workout specialist agent.
    Use for: exercise plans, sets and reps, form advice, training schedule,
    rest days, injury modifications to workout plan, fitness progress.
    Do NOT use for diet or nutrition questions.

    Args:
        message: The user's workout-related question or request,
                 including any relevant context.

    Returns:
        The workout agent's response as a string.
    """
    return _run_subagent(_get_workout_graph(), message, WORKOUT_THREAD)


@tool
def ask_dietary_agent(message: str) -> str:
    """
    Send a diet or nutrition related question to the dietary specialist agent.
    Use for: meal plans, caloric targets, macros, food choices, meal timing,
    dietary restrictions, food allergies, diet progress tracking.
    Do NOT use for workout or exercise questions.

    Args:
        message: The user's diet-related question or request,
                 including any relevant context.

    Returns:
        The dietary agent's response as a string.
    """
    return _run_subagent(_get_diet_graph(), message, DIET_THREAD)