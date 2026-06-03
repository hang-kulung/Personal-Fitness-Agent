from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from supervisor.state import SupervisorState
from supervisor.nodes.agent import build_agent_node
from supervisor.nodes.profile import load_profile_node, save_profile_node
from supervisor.nodes.subagent_tools import ask_workout_agent, ask_dietary_agent

ALL_TOOLS = [ask_workout_agent, ask_dietary_agent]


def build_graph(checkpointer):
    agent_node = build_agent_node(ALL_TOOLS)
    tools_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(SupervisorState)

    # ── nodes ─────────────────────────────────────────────────────────────
    graph.add_node("load_profile",  load_profile_node)
    graph.add_node("agent",         agent_node)
    graph.add_node("tools",         tools_node)
    graph.add_node("save_profile",  save_profile_node)

    # ── edges ─────────────────────────────────────────────────────────────
    graph.add_edge(START,          "load_profile")
    graph.add_edge("load_profile", "agent")

    graph.add_conditional_edges("agent", tools_condition, {
        "tools": "tools",
        END:     "save_profile",
    })

    graph.add_edge("tools",        "agent")
    graph.add_edge("save_profile", END)

    return graph.compile(checkpointer=checkpointer)