import json
import os
import tempfile
from datetime import datetime
from langchain_core.messages import AIMessage

MEMORY_DIR = "../shared/memory"
MAX_RECENT_EVENTS = 20


def _read(filename: str, default):
    p = os.path.join(MEMORY_DIR, filename)
    if not os.path.exists(p):
        return default
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def load_memory_node(state: dict) -> dict:
    """
    Runs BEFORE the agent. Loads all diet memory + shared user profile.
    user_profile.json is read-only here — never written by diet agent.
    """
    from tools.meal_manager import _load_plan

    now = datetime.now()
    return {
        "user_profile":  _read("user_profile.json",    {}),  # read-only
        "diet_prefs":    _read("diet_preferences.json", {}),
        "diet_progress": _read("diet_progress.json",   []),
        "meal_plan":     _load_plan(),
        "today": {
            "current_date":  now.strftime("%Y-%m-%d"),
            "weekday":       now.strftime("%A"),
            "weekday_index": now.weekday(),
        },
    }


def save_memory_node(state: dict) -> dict:
    """
    Runs AFTER the agent. Scans last AI messages for notable diet events
    and writes them to diet_events.json (ring buffer, max 20 entries).
    """
    EVENT_RULES = [
        ("weight",      ["lost weight", "gained weight", "new weight", "weigh"]),
        ("allergy",     ["allergy", "allergic", "intolerance", "avoid"]),
        ("preference",  ["don't like", "hate", "love", "prefer", "favourite"]),
        ("adherence",   ["skipped meal", "missed", "cheat meal", "ate out"]),
        ("milestone",   ["goal reached", "milestone", "achievement"]),
    ]

    messages  = state.get("messages", [])
    ai_messages = [
        m for m in messages[-10:]
        if isinstance(m, AIMessage) and isinstance(m.content, str)
    ]

    events_path = os.path.join(MEMORY_DIR, "diet_events.json")
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
                break

    capped = events[-MAX_RECENT_EVENTS:]
    with tempfile.NamedTemporaryFile(
        "w", dir=MEMORY_DIR, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(capped, tmp, indent=2)
    os.replace(tmp.name, events_path)

    return {}