import json, os, tempfile
from datetime import datetime
from langchain_core.tools import tool

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, "../.."))

MEMORY_DIR = os.path.join(_ROOT, "shared", "memory")

def _mem_path(filename: str) -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return os.path.join(MEMORY_DIR, filename)


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: str, data) -> None:
    """Atomic write."""
    directory = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        "w", dir=directory, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

# ════════════════════════════════════════════════════════════════════════════════
# NEW: PROFILE / PREFERENCE / PROGRESS TOOLS
# These write to memory/ so the memory service can surface them efficiently.
# ════════════════════════════════════════════════════════════════════════════════

@tool
def update_user_profile(updates: dict) -> dict:
    """
    Persist key facts about the user that should survive across sessions.
    Call this when the user shares or changes: age, fitness level, equipment,
    injuries, or their primary goal.

    This data is always injected into memory context at the start of each session,
    so the agent never needs to re-ask for information already recorded here.

    Args:
        updates: Dict with any of:
            age (int), fitness_level (str), equipment (list[str]),
            injuries (list[str]), goal (str), notes (str)

        Example:
            {"age": 28, "fitness_level": "intermediate",
             "equipment": ["dumbbells", "pull-up bar"],
             "injuries": ["mild lower-back pain"], "goal": "muscle gain"}

    Returns:
        The full updated profile.
    """
    p = _mem_path("user_profile.json")
    profile = _read_json(p, default={})
    profile.update(updates)
    profile["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    _write_json(p, profile)
    return {"status": "profile_updated", "profile": profile}


@tool
def update_preferences(updates: dict) -> dict:
    """
    Record the user's workout preferences so future sessions respect them.
    Call this when the user expresses likes, dislikes, or style preferences.

    Args:
        updates: Dict with any of:
            liked_exercises (list[str]),  disliked_exercises (list[str]),
            preferred_workout_duration_minutes (int),
            rest_day_preference (list[str] — e.g. ["Sunday"]),
            notes (str)

        Example:
            {"liked_exercises": ["pull-ups", "deadlifts"],
             "disliked_exercises": ["burpees"],
             "preferred_workout_duration_minutes": 45}

    Returns:
        The full updated preferences dict.
    """
    p = _mem_path("preferences.json")
    prefs = _read_json(p, default={})

    # Merge lists rather than overwrite them
    for list_key in ("liked_exercises", "disliked_exercises", "rest_day_preference"):
        if list_key in updates:
            existing = set(prefs.get(list_key, []))
            existing.update(updates.pop(list_key))
            prefs[list_key] = sorted(existing)

    prefs.update(updates)
    prefs["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    _write_json(p, prefs)
    return {"status": "preferences_updated", "preferences": prefs}


@tool
def log_progress(entry: str) -> dict:
    """
    Append a permanent progress milestone — PRs, body-weight changes,
    significant goal achievements.  Unlike recent_events (which is a
    capped ring buffer), this log is kept forever but should only contain
    genuinely significant milestones.

    Args:
        entry: One-line description of the milestone.
               Example: "Deadlifted 120 kg for the first time (5 reps)."

    Returns:
        Confirmation with the full progress log.
    """
    p = _mem_path("progress_log.json")
    log: list = _read_json(p, default=[])
    log.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "text": entry,
    })
    _write_json(p, log)
    return {"status": "logged", "progress_log": log}
