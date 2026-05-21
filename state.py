# from typing import Annotated, Optional
from langgraph.graph import MessagesState


class WorkoutState(MessagesState):
    """
    Flows through every node in the graph.
    MessagesState provides: messages: Annotated[list, add_messages]
    We add workout-specific fields on top.
    """

    # ── session identity ──────────────────────────────────────────────────
    user_id:   str           # e.g. "user_001"  — used to namespace memory store
    thread_id: str           # e.g. "user_001_session_1" — used by checkpointer

    # ── populated by load_memory_node at session start ────────────────────
    user_profile:  dict      # age, fitness level, injuries, equipment, goal
    preferences:   dict      # liked/disliked exercises, duration prefs
    progress_log:  list      # PRs and milestones

    # ── populated by agent_node or tools on first call ────────────────────
    today:         dict      # {current_date, weekday, weekday_index}
    workout_plan:  dict      # the full 7-day plan from workout_plan.json