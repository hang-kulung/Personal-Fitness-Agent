import json
import os
import tempfile
from datetime import datetime
from langchain_core.tools import tool

# ── file paths ────────────────────────────────────────────────────────────────

PLAN_FILE    = "../data/meal_plan.json"
MEMORY_DIR   = "../shared/memory"


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
    if not os.path.exists(PLAN_FILE):
        return {}
    try:
        with open(PLAN_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    
def _save_plan(plan: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=".", delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(plan, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, PLAN_FILE)

# ════════════════════════════════════════════════════════════════════════════════
# PLAN TOOLS  
# ════════════════════════════════════════════════════════════════════════════════

@tool
def save_meal_plan(plan: dict) -> dict:
    """
    Save the full 7-day meal plan to persistent storage.
    Call this ONCE after creating a new plan. Not for small adjustments.

    Args:
        plan: Dict with this structure:
            {
              "created_at": "YYYY-MM-DD",
              "goal": "e.g. muscle gain",
              "daily_calories": 2800,
              "macros": {
                "protein_g": 180,
                "carbs_g": 300,
                "fat_g": 80
              },
              "days": {
                "Monday": {
                  "type": "training_day",
                  "total_calories": 2800,
                  "meals": [
                    {
                      "name": "Breakfast",
                      "time": "8:00 AM",
                      "foods": [
                        {"item": "Oats", "amount": "80g", "calories": 300},
                        {"item": "Banana", "amount": "1 medium", "calories": 90}
                      ],
                      "total_calories": 390,
                      "macros": {"protein_g": 10, "carbs_g": 70, "fat_g": 5}
                    }
                  ]
                },
                "Tuesday": { "type": "rest_day", ... },
                ...
              },
              "notes": "any extra notes"
            }

    Returns:
        Confirmation with status, version, saved_at.
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
def get_meal_plan() -> dict:
    """
    Load the full 7-day meal plan from persistent storage.
    Call this at the start of every session to read the current plan.

    Returns:
        The full meal plan dict if it exists, or {"status": "no_plan"} if none saved yet.
    """
    plan = _load_plan()
    return plan if plan else {"status": "no_plan"}


@tool
def get_todays_meals(weekday: str) -> dict:
    """
    Get only today's meals from the saved plan.

    Args:
        weekday: Full weekday name e.g. 'Monday', 'Thursday'.

    Returns:
        Dict with today's meals and macros,
        or {"status": "no_plan"} / {"status": "day_not_found"}.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "no_plan"}

    today = plan.get("days", {}).get(weekday)
    if not today:
        return {"status": "day_not_found", "weekday": weekday}

    return {
        "status":          "ok",
        "weekday":         weekday,
        "type":            today.get("type", "training_day"),
        "total_calories":  today.get("total_calories"),
        "meals":           today.get("meals", []),
    }


@tool
def update_meal(weekday: str, meal_name: str, updated_meal: dict) -> dict:
    """
    Update a single meal within a day.
    Use for small changes: swap a food item, adjust portions, change timing.

    Args:
        weekday:      Full weekday name e.g. 'Monday'.
        meal_name:    Meal to update e.g. 'Breakfast', 'Lunch' (case-insensitive).
        updated_meal: Dict with any of: name, time, foods, total_calories, macros.

    Returns:
        Confirmation or error if meal not found.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "error", "message": "No plan exists yet."}

    day = plan["days"].get(weekday)
    if not day:
        return {"status": "error", "message": f"Day '{weekday}' not found."}

    meals = day.get("meals", [])
    for i, meal in enumerate(meals):
        if meal.get("name", "").lower() == meal_name.lower():
            meals[i].update(updated_meal)
            plan["days"][weekday]["meals"] = meals
            plan["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            plan["version"]       = plan.get("version", 0) + 1
            _save_plan(plan)
            return {
                "status":  "updated",
                "weekday": weekday,
                "meal":    meals[i],
                "version": plan["version"],
            }

    return {
        "status":    "error",
        "message":   f"Meal '{meal_name}' not found on {weekday}.",
        "available": [m["name"] for m in meals],
    }


@tool
def update_day_nutrition(weekday: str, updated_day: dict) -> dict:
    """
    Replace the full nutrition plan for a single day.
    Use when the user's schedule changes or a full day needs restructuring.

    Args:
        weekday:     Full weekday name e.g. 'Wednesday'.
        updated_day: New day dict — keys: type, total_calories, meals.

    Returns:
        Confirmation with updated day content.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "error", "message": "No plan exists yet."}

    plan["days"][weekday] = updated_day
    plan["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    plan["version"]       = plan.get("version", 0) + 1
    _save_plan(plan)
    return {
        "status":      "updated",
        "weekday":     weekday,
        "new_content": updated_day,
        "version":     plan["version"],
    }




@tool
def add_diet_note(note: str) -> dict:
    """
    Append a note to the meal plan log.
    Use for session observations that don't fit structured tools.
    For food preferences or allergies, prefer update_diet_preferences instead.

    Args:
        note: Free text note.

    Returns:
        Confirmation with all current notes.
    """
    plan = _load_plan()
    if not plan:
        return {"status": "error", "message": "No plan exists yet."}

    notes_log = plan.get("notes_log", [])
    notes_log.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "note": note,
    })
    plan["notes_log"] = notes_log
    _save_plan(plan)
    return {"status": "noted", "all_notes": notes_log}
