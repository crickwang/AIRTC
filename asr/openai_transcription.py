import os
import requests
from dotenv import load_dotenv
import yaml
import json

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MODEL = config.get('openai_model')
load_dotenv()

API = os.environ.get("OPENAI_API_KEY")
URL = config.get('openai_transcription_base_url')
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

class OpenAITranscription:
    """
    A class to handle transcription using OpenAI's API.
    """
    def __init__(self, model_name: str = MODEL) -> None:
        """
        Initialize the OpenAITranscription class.
        Args:
            model_name (str, optional): The name of the OpenAI model to use for transcription.
                Defaults to the model specified in the config file.
        """
        self.model_name = model_name

    def generate(self, speech: str) -> str:
        """
        Generate a transcription for the given speech audio file.

        Args:
            speech (str): The path to the audio file to transcribe.

        Returns:
            str: The transcription result.
        """
        if not API:
            return json.dumps({"error": "OPENAI_API_KEY not set"}), 500

        files = {
            "file": open(speech, 'rb'),
        }
        headers = {
            "Authorization": f"Bearer {API}",
        }
        
        data = {
            "model": self.model_name,
        }

        try:
            response = requests.post(
                url=URL,
                headers=headers,
                data=data,
                files=files
            )
            print(response.json())
            return response.json().get('text', 'No transcription found')
        except requests.RequestException as e:
            return json.dumps({"error": str(e)}), 500

if __name__ == "__main__":
    app = OpenAITranscription()
    app.generate(os.path.join(PARENT_DIR, "audio/english.wav"))
