import json
import time

from aiortc import RTCDataChannel


def get_current_time() -> int:
    """Get the current time in milliseconds."""
    return int(round(time.time() * 1000))

def server_to_client(dc: RTCDataChannel, msg: str, msg_type: str = "log"):
    """
    Send a message to the client via the WebRTC data channel.
    Args:
        dc (RTCDataChannel): The data channel to send the message.
        msg (str): The message to send.
        msg_type (str): The message type ("log" or "transcription"). Defaults to "log".
    Exception:
        If the data channel is not open or an error occurs while sending the message.
    """
    if dc and dc.readyState == "open":
        try:
            log_data = json.dumps({"type": msg_type, "message": msg})
            dc.send(log_data)
        except Exception as e:
            print(f"Error sending log to client: {e}")
