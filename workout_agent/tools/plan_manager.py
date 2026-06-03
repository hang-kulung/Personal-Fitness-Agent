import json
import os
import tempfile
from datetime import datetime
from langchain_core.tools import tool

# ── file paths ────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, "../.."))   # Project root

PLAN_FILE  = os.path.join(_ROOT, "data", "workout_plan.json")
MEMORY_DIR = os.path.join(_ROOT, "shared", "memory")

# ── generic helpers ───────────────────────────────────────────────────────────

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


# ── plan helpers (unchanged) ──────────────────────────────────────────────────

def _load_plan() -> dict:
    return _read_json(PLAN_FILE, default={})


def _save_plan(plan: dict) -> None:
    _write_json(PLAN_FILE, plan)


# ════════════════════════════════════════════════════════════════════════════════
# PLAN TOOLS  
# ════════════════════════════════════════════════════════════════════════════════

@tool
def save_workout_plan(plan: dict) -> dict:
    """
    Save the full 7-day workout plan to persistent storage.
    Call this ONCE after creating a brand-new plan for the user.
    Do NOT call this for small adjustments — use update_exercise or update_day.

    Args:
        plan: A dict with this exact structure:
            {
              "created_at": "YYYY-MM-DD",
              "goal": "e.g. muscle gain",
              "fitness_level": "beginner | intermediate | advanced",
              "days": {
                "Monday":    { "focus": "...", "exercises": [...] },
                "Tuesday":   { "focus": "Rest", "exercises": [] },
                "Wednesday": { "focus": "...", "exercises": [...] },
                "Thursday":  { "focus": "...", "exercises": [...] },
                "Friday":    { "focus": "...", "exercises": [...] },
                "Saturday":  { "focus": "...", "exercises": [...] },
                "Sunday":    { "focus": "Rest", "exercises": [] }
              },
              "notes": "any extra notes"
            }
            Each exercise in the list should be a dict with keys:
              name, sets, reps_or_duration, rest_seconds, form_cue

    Returns:
        Confirmation dict with status and saved_at timestamp.
    """
    plan["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    plan["version"]  = _load_plan().get("version", 0) + 1
    _save_plan(plan)
    return {
        "status":   "saved",
        "version":  plan["version"],
        "saved_at": plan["saved_at"],
    }


@tool
def get_workout_plan() -> dict:
    """
    Load the full 7-day workout plan from persistent storage.
    Call this at the start of every session to read the current plan.

    Returns:
        The full plan dict if it exists, or {"status": "no_plan"} if none saved yet.
    """
    plan = _load_plan()
    return plan if plan else {"status": "no_plan"}


@tool
def get_todays_workout(weekday: str) -> dict:
    """
    Get only today's workout from the saved plan.

    Args:
        weekday: Full weekday name e.g. 'Monday', 'Tuesday', etc.

    Returns:
        Dict with today's focus and exercises,
        or {"status": "no_plan"} / {"status": "rest_day"}.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "no_plan"}

    today = plan.get("days", {}).get(weekday)
    if not today:
        return {"status": "day_not_found", "weekday": weekday}

    if today.get("focus", "").lower() == "rest":
        return {
            "status":  "rest_day",
            "weekday": weekday,
            "message": "Today is a rest day. Light stretching is fine.",
        }

    return {
        "status":    "ok",
        "weekday":   weekday,
        "focus":     today["focus"],
        "exercises": today["exercises"],
    }


@tool
def update_day(weekday: str, updated_day: dict) -> dict:
    """
    Replace the workout for a single day.
    Use when a full day needs restructuring (e.g. new injury changes Wednesday).

    Args:
        weekday:     Full weekday name e.g. 'Wednesday'.
        updated_day: New day dict — keys: focus (str), exercises (list).
                     Each exercise: name, sets, reps_or_duration, rest_seconds, form_cue.

    Returns:
        Confirmation with the updated day content.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "error", "message": "No plan exists yet. Create one first."}

    plan["days"][weekday]  = updated_day
    plan["last_modified"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
    plan["version"]        = plan.get("version", 0) + 1
    _save_plan(plan)
    return {
        "status":      "updated",
        "weekday":     weekday,
        "new_content": updated_day,
        "version":     plan["version"],
    }


@tool
def update_exercise(weekday: str, exercise_name: str, updated_exercise: dict) -> dict:
    """
    Update a single exercise within a day.
    Use for small tweaks: reducing sets, swapping an exercise, adjusting rest time.

    Args:
        weekday:          Full weekday name e.g. 'Monday'.
        exercise_name:    Exact name of the exercise (case-insensitive match).
        updated_exercise: Dict with any subset of:
                          name, sets, reps_or_duration, rest_seconds, form_cue.

    Returns:
        Confirmation or error if exercise not found.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "error", "message": "No plan exists yet."}

    day = plan["days"].get(weekday)
    if not day:
        return {"status": "error", "message": f"Day '{weekday}' not found in plan."}

    exercises = day.get("exercises", [])
    for i, ex in enumerate(exercises):
        if ex.get("name", "").lower() == exercise_name.lower():
            exercises[i].update(updated_exercise)
            plan["days"][weekday]["exercises"] = exercises
            plan["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            plan["version"]       = plan.get("version", 0) + 1
            _save_plan(plan)
            return {
                "status":   "updated",
                "weekday":  weekday,
                "exercise": exercises[i],
                "version":  plan["version"],
            }

    return {
        "status":    "error",
        "message":   f"Exercise '{exercise_name}' not found on {weekday}.",
        "available": [e["name"] for e in exercises],
    }


@tool
def add_plan_note(note: str) -> dict:
    """
    Append a short note to the plan log — use for session observations that
    don't fit the structured tools (e.g. 'user wants more variety next week').
    For injuries, PRs, or goal changes, prefer update_user_profile or log_progress.

    Args:
        note: Free text note.

    Returns:
        Confirmation with all current notes.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "error", "message": "No plan exists yet."}

    notes_log = plan.get("notes_log", [])
    notes_log.append({"date": datetime.now().strftime("%Y-%m-%d"), "note": note})
    plan["notes_log"] = notes_log
    _save_plan(plan)
    return {"status": "noted", "all_notes": notes_log}
