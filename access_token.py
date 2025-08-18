
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
client_id = os.getenv('BAIDU_API_KEY')
client_secret = os.getenv('BAIDU_SECRET_KEY')

def baidu():
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()['access_token']

import os

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

PROJECT_ID = os.getenv("GOOGLE_PROJ_ID")


def create_recognizer(recognizer_id: str) -> cloud_speech.Recognizer:
    """Сreates a recognizer with an unique ID and default recognition configuration.
    Args:
        recognizer_id (str): The unique identifier for the recognizer to be created.
    Returns:
        cloud_speech.Recognizer: The created recognizer object with configuration.
    """
    # Instantiates a client
    client = SpeechClient()

    request = cloud_speech.CreateRecognizerRequest(
        parent=f"projects/{PROJECT_ID}/locations/global",
        recognizer_id=recognizer_id,
        recognizer=cloud_speech.Recognizer(
            default_recognition_config=cloud_speech.RecognitionConfig(
                language_codes=["en-US"], model="long"
            ),
        ),
    )
    # Sends the request to create a recognizer and waits for the operation to complete
    operation = client.create_recognizer(request=request)
    recognizer = operation.result()

    print("Created Recognizer:", recognizer.name)
    return recognizer

if __name__ == '__main__':
    # baidu()
    create_recognizer("myrecognizer")