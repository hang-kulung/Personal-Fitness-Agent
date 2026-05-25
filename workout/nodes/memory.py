import json
import os
from datetime import datetime
from langchain_core.messages import AIMessage

MEMORY_DIR = "memory"

def _read(filename, default):
    p = os.path.join(MEMORY_DIR, filename)
    if not os.path.exists(p):
        return default
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def load_memory_node(state: dict) -> dict:
    """
    Runs BEFORE the agent. Reads all structured memory files and the
    workout plan, injects them into state so the agent prompt can use them.
    """
    from tools.plan_manager import _load_plan   # reuse existing helper
    from datetime import datetime

    now = datetime.now()
    return {
        "user_profile": _read("user_profile.json",  {}),
        "preferences":  _read("preferences.json",   {}),
        "progress_log": _read("progress_log.json",  []),
        "workout_plan": _load_plan(),
        "today": {
            "current_date":  now.strftime("%Y-%m-%d"),
            "weekday":       now.strftime("%A"),
            "weekday_index": now.weekday(),
        },
    }


def save_memory_node(state: dict) -> dict:
    """
    Runs AFTER the agent finishes (no more tool calls).
    Scans the last few AI messages for notable facts and writes
    them to recent_events.json (ring buffer, max 20 entries).
    """
    import tempfile, json

    EVENT_RULES = [
        ("pr",       ["new pr", "personal record", "personal best"]),
        ("injury",   ["injury", "pain", "discomfort", "avoid"]),
        ("feedback", ["too easy", "too hard", "increase", "decrease", "adjust"]),
        ("skipped",  ["skipped", "missed", "rest day"]),
        ("goal",     ["new goal", "change goal", "goal is now"]),
    ]

    messages = state.get("messages", [])
    # only scan AI messages from this session (last 10 to be safe)
    ai_messages = [
        m for m in messages[-10:]
        if isinstance(m, AIMessage) and isinstance(m.content, str)
    ]

    events_path = os.path.join(MEMORY_DIR, "recent_events.json")
    os.makedirs(MEMORY_DIR, exist_ok=True)

    try:
        with open(events_path) as f:
            events = json.load(f)
    except Exception:
        events = []

    today = datetime.now().strftime("%Y-%m-%d")
    for msg in ai_messages:
        tl = msg.content.lower()
        for event_type, keywords in EVENT_RULES:
            if any(kw in tl for kw in keywords):
                events.append({
                    "date": today,
                    "type": event_type,
                    "text": msg.content[:300],
                })
                break   # one event per message

    # ring buffer cap
    capped = events[-20:]
    with tempfile.NamedTemporaryFile(
        "w", dir=MEMORY_DIR, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(capped, tmp, indent=2)
    os.replace(tmp.name, events_path)

    return {}   # no state update needed