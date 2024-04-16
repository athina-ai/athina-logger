from datetime import datetime, timezone


def remove_none_values(d):
    if isinstance(d, dict):
        return {k: remove_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [remove_none_values(v) for v in d if v is not None]
    else:
        return d


def get_utc_end_time(end_time):
    if end_time is not None:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        end_time = end_time.astimezone(timezone.utc)
    else:
        end_time = datetime.now(timezone.utc)      
    return end_time