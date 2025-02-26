from datetime import datetime, timezone
import json

def remove_none_values(d):
    if isinstance(d, dict):
        return {k: remove_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_none_values(v) for v in d if v is not None]
    else:
        return d

def sanitize_dict(d):
    """Recursively sanitize dictionary to ensure JSON serializability."""
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(item) for item in d]
    elif hasattr(d, '__dict__'):  # Custom object
        return f"<{d.__class__.__name__} object>"
    else:
        # Handle basic types that are JSON serializable
        try:
            json.dumps(d)
            return d
        except (TypeError, OverflowError):
            return str(d)  # Convert to string as fallback

def get_utc_time(time_obj=None):
    if time_obj is None:
        return datetime.now(timezone.utc)
    if time_obj.tzinfo is None:
        time_obj = time_obj.replace(tzinfo=timezone.utc)
    return time_obj.astimezone(timezone.utc)
