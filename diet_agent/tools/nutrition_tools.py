import json
import os
import tempfile
from datetime import datetime
from langchain_core.tools import tool

MEMORY_DIR = "../shared/memory"


def _path(filename: str) -> str:
    return os.path.join(MEMORY_DIR, filename)


def _read(filename: str, default):
    p = _path(filename)
    if not os.path.exists(p):
        return default
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write(filename: str, data) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    p = _path(filename)
    with tempfile.NamedTemporaryFile(
        "w", dir=MEMORY_DIR, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(data, tmp, indent=2)
    os.replace(tmp.name, p)


@tool
def update_diet_preferences(updates: dict) -> dict:
    """
    Record the user's dietary preferences and restrictions.
    Call when the user mentions food likes, dislikes, allergies, or restrictions.

    Args:
        updates: Dict with any of:
            allergies (list[str])         e.g. ["peanuts", "shellfish"]
            restrictions (list[str])      e.g. ["vegetarian", "halal", "gluten-free"]
            disliked_foods (list[str])    e.g. ["broccoli", "liver"]
            liked_foods (list[str])       e.g. ["chicken", "rice", "eggs"]
            meals_per_day (int)           e.g. 4
            notes (str)

    Returns:
        The full updated preferences dict.
    """
    prefs = _read("diet_preferences.json", {})

    # merge lists rather than overwrite
    for list_key in ("allergies", "restrictions", "disliked_foods", "liked_foods"):
        if list_key in updates:
            existing = set(prefs.get(list_key, []))
            existing.update(updates.pop(list_key))
            prefs[list_key] = sorted(existing)

    prefs.update(updates)
    prefs["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    _write("diet_preferences.json", prefs)
    return {"status": "preferences_updated", "preferences": prefs}


@tool
def log_diet_progress(entry: str) -> dict:
    """
    Append a permanent diet/body progress milestone.
    Use for weight changes, measurements, or notable adherence achievements.
    Example: "Reached 75kg bodyweight. Down 2kg from start."

    Args:
        entry: One-line description of the milestone.

    Returns:
        Confirmation with the full progress log.
    """
    log = _read("diet_progress.json", [])
    log.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "text": entry,
    })
    _write("diet_progress.json", log)
    return {"status": "logged", "diet_progress": log}