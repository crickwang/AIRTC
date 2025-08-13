import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_REGION = "eastus"
# Azure TTS Voice Gallery: [https://speech.microsoft.com/portal/voicegallery]
AZURE_TTS_VOICE = "zh-CN-YunyangNeural"

# Initialize Azure TTS globally (add this near your other model initializations)
azure_speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_TTS_KEY,
    region=AZURE_TTS_REGION
)
azure_speech_config.speech_synthesis_voice_name = AZURE_TTS_VOICE
azure_speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
)

# Pre-create synthesizer and connection for lower latency
azure_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=azure_speech_config,
    audio_config=None  # We handle audio stream manually
)
azure_connection = speechsdk.Connection.from_speech_synthesizer(azure_synthesizer)
azure_connection.open(True)  # Pre-connect

result = azure_synthesizer.speak_text("Hello, this is a test.")
print(result.cancellation_details.error_details)
