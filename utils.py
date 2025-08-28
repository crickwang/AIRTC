import argparse
import ssl
from clients.ASR.model import ASRClientFactory
from clients.LLM.model import LLMClientFactory
from clients.TTS.model import TTSClientFactory
from vad.vad import VADFactory
import requests
from dotenv import load_dotenv
import os
import json
from aiortc import RTCDataChannel
# import boto3
# from botocore.exceptions import ClientError

load_dotenv()

def get_ssl_context(args):
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
        return ssl_context
    return None

def create_client(model_type, platform, **kwargs):
    """
    Create a client (ASR, LLM, TTS, VAD) for the specified model type and platform.
    Args:
        model_type (str): The type of model (one of 'asr', 'llm', 'tts', 'vad').
        platform (str): The platform for the model (e.g., "baidu", "google").
                        See the register to get all possible platforms.
        **kwargs: Additional keyword arguments for client initialization.
    """
    try:
        if model_type == "asr":
            return ASRClientFactory.create(platform, **kwargs)
        elif model_type == "llm":
            return LLMClientFactory.create(platform, **kwargs)
        elif model_type == "tts":
            return TTSClientFactory.create(platform, **kwargs)
        elif model_type == "vad":
            return VADFactory.create(platform, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f"Error creating client: {e}")
        return None
    
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
    print(response.json()['access_token'])
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

def log_to_client(dc: RTCDataChannel, msg: str):
    """
    Log a message to the WebRTC data channel.
    Args:
        dc (RTCDataChannel): The WebRTC data channel.
        msg (str): The message to log.
    """
    print("get message", msg, dc.readyState)
    if dc and (dc.readyState == "open"):
        try:
            log_data = json.dumps({"type": "log", "message": msg})
            dc.send(log_data)
        except Exception as e:
            print(f"Error sending log to client: {e}")

def get_secret():
    secret_name = "AIRTC"
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']

if __name__ == "__main__":
    baidu()


