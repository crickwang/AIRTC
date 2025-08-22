import time 
import json
from aiortc import RTCDataChannel

def get_current_time() -> int:
    """Get the current time in milliseconds."""
    return int(round(time.time() * 1000))

def log_to_client(dc: RTCDataChannel, msg: str):
    if dc and dc.readyState == "open":
        try:
            log_data = json.dumps({"type": "log", "message": msg})
            dc.send(log_data)
        except Exception as e:
            print(f"Error sending log to client: {e}")