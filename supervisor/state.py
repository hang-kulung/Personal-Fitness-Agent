from langgraph.graph import MessagesState


class SupervisorState(MessagesState):
    """
    State for the supervisor graph.
    MessagesState provides: messages: Annotated[list, add_messages]
    """

    # ── session identity ──────────────────────────────────────────────────
    user_id:   str
    thread_id: str

    # ── owned by supervisor — subagents read, never write ─────────────────
    user_profile: dict    # age, fitness_level, goal, injuries, equipment

    # ── populated by load_profile_node ───────────────────────────────────
    today: dict           # {current_date, weekday, weekday_index}

    # ── routing metadata (optional, useful for debugging) ────────────────
    last_routed_to: str   # "workout", "diet", "both", or "none"