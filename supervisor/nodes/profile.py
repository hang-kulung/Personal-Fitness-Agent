import json
import os
import tempfile
from datetime import datetime

_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.abspath(os.path.join(_HERE, "../.."))

SHARED_MEMORY = os.path.join(_ROOT, "shared", "memory")
PROFILE_FILE  = "user_profile.json"

def _profile_path() -> str:
    return os.path.join(SHARED_MEMORY, PROFILE_FILE)


def _read_profile() -> dict:
    p = _profile_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_profile(profile: dict) -> None:
    os.makedirs(SHARED_MEMORY, exist_ok=True)
    p = _profile_path()
    with tempfile.NamedTemporaryFile(
        "w", dir=SHARED_MEMORY, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(profile, tmp, indent=2)
    os.replace(tmp.name, p)


def load_profile_node(state: dict) -> dict:
    """
    Runs BEFORE the supervisor agent.
    Loads user_profile.json and today's date into state.
    """
    now = datetime.now()
    return {
        "user_profile": _read_profile(),
        "today": {
            "current_date":  now.strftime("%Y-%m-%d"),
            "weekday":       now.strftime("%A"),
            "weekday_index": now.weekday(),
        },
    }


def save_profile_node(state: dict) -> dict:
    """
    Runs AFTER the supervisor agent finishes.
    Writes user_profile back to disk if it was updated during the session.
    """
    profile = state.get("user_profile", {})
    if profile:
        profile["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        _write_profile(profile)
    return {}