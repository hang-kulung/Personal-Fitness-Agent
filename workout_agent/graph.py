from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from workout_agent.state import WorkoutState
from workout_agent.nodes.agent import build_agent_node
from workout_agent.nodes.memory import load_memory_node, save_memory_node
from workout_agent.tools.date_tool import get_current_date
from workout_agent.tools.web_search import web_search
from workout_agent.tools.memory_tools import update_user_profile, update_preferences, log_progress
from workout_agent.tools.plan_manager import (
    save_workout_plan,
    get_workout_plan,
    get_todays_workout,
    update_day,
    update_exercise,
    add_plan_note,
)

ALL_TOOLS = [
    get_current_date,
    web_search,
    update_user_profile,
    update_preferences,
    log_progress,
    save_workout_plan,
    get_workout_plan,
    get_todays_workout,
    update_day,
    update_exercise,
    add_plan_note,
]


def build_graph(checkpointer):
    agent_node = build_agent_node(ALL_TOOLS)
    tools_node = ToolNode(ALL_TOOLS)   # handles all tool execution automatically

    graph = StateGraph(WorkoutState)

    # ── nodes ─────────────────────────────────────────────────────────────
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("agent",       agent_node)
    graph.add_node("tools",       tools_node)
    graph.add_node("save_memory", save_memory_node)

    # ── edges ─────────────────────────────────────────────────────────────
    graph.add_edge(START,         "load_memory")
    graph.add_edge("load_memory", "agent")

    # tools_condition: if the model's last message has tool_calls → "tools"
    #                  otherwise → "save_memory"
    graph.add_conditional_edges("agent", tools_condition, {
        "tools": "tools",
        END:     "save_memory",
    })

    graph.add_edge("tools",       "agent")       # loop back after tool execution
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer)