from datetime import datetime, timezone


def remove_none_values(d):
    if isinstance(d, dict):
        return {k: remove_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_none_values(v) for v in d if v is not None]
    else:
        return d

def get_utc_time(time_obj=None):
    if time_obj is None:
        return datetime.now(timezone.utc)
    if time_obj.tzinfo is None:
        time_obj = time_obj.replace(tzinfo=timezone.utc)
    return time_obj.astimezone(timezone.utc)
