import time 
import json
from aiortc import RTCDataChannel

def get_current_time() -> int:
    """Get the current time in milliseconds."""
    return int(round(time.time() * 1000))

def server_to_client(dc: RTCDataChannel, msg: str):
    """
    Log a message to the client via the data channel of WebRTC.
    Args:
        dc (RTCDataChannel): The data channel to send the log message.
        msg (str): The log message to send.
    Exception:
        If the data channel is not open or an error occurs while sending the log message.
    """
    if dc and dc.readyState == "open":
        try:
            log_data = json.dumps({"type": "log", "message": msg})
            dc.send(log_data)
        except Exception as e:
            print(f"Error sending log to client: {e}")