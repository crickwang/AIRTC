import time 
def get_current_time() -> int:
    """Get the current time in milliseconds."""
    return int(round(time.time() * 1000))