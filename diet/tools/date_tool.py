from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_date() -> dict:
    """
    Returns today's date and weekday.
    Use this to know what day to plan meal on.

    Returns:
        current_date: YYYY-MM-DD string
        weekday: full name e.g. 'Monday'
        weekday_index: 0=Monday … 6=Sunday
    """
    now = datetime.now()
    return {
        "current_date":  now.strftime("%Y-%m-%d"),
        "weekday":       now.strftime("%A"),
        "weekday_index": now.weekday(),
    }