from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from state import DietState
from nodes.agent import build_agent_node
from nodes.memory import load_memory_node, save_memory_node

from tools.date_tool       import get_current_date
from tools.web_search      import web_search
from tools.nutrition_tools import update_diet_preferences, log_diet_progress
from tools.meal_manager    import (
    save_meal_plan,
    get_meal_plan,
    get_todays_meals,
    update_meal,
    update_day_nutrition,
    add_diet_note,
)

ALL_TOOLS = [
    get_current_date,
    web_search,
    update_diet_preferences,
    log_diet_progress,
    save_meal_plan,
    get_meal_plan,
    get_todays_meals,
    update_meal,
    update_day_nutrition,
    add_diet_note,
]


def build_graph(checkpointer):
    agent_node = build_agent_node(ALL_TOOLS)
    tools_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(DietState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("agent",       agent_node)
    graph.add_node("tools",       tools_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START,         "load_memory")
    graph.add_edge("load_memory", "agent")

    graph.add_conditional_edges("agent", tools_condition, {
        "tools": "tools",
        END:     "save_memory",
    })

    graph.add_edge("tools",       "agent")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=checkpointer)