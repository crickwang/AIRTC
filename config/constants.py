from config.settings import Settings
import os

# Load configuration from YAML file;
# Load environment variables and private tokens from .env file
settings = Settings()

# Uncomment and set your Google Cloud credentials if needed
# Follow google official documentation to set up your credentials
# [https://cloud.google.com/docs/authentication/getting-started]
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_.json_credential_file"
# os.environ["GOOGLE_CLOUD_PROJECT"] = your_project_id

# Environment Variables and Constants
TIME_PER_CHUNK = 10  # in ms
ASR_SAMPLE_RATE = 24000  # in Hz
ASR_LANGUAGE = settings.get_config("asr_language", "zh-CN")  # Language code, please refer to
                         # [https://developers.google.com/workspace/admin/directory/v1/languages]
ASR_CHUNK_SIZE = int(ASR_SAMPLE_RATE * TIME_PER_CHUNK / 1000)
CHANNELS = 1
FORMAT = "s16"
LAYOUT = "mono"  # Audio layout, mono or stereo
SILENCE_TIME = 1  # in seconds
SILENCE_CHUNK = 5
STREAMING_LIMIT = 60000  # 1 minute, in ms

# Constants for local ASR 
FUN_ASR_MODEL = settings.get_config("funasr_model")
ASR_ENCODER_CHUNK_LOOK_BACK = 4
ASR_DECODER_CHUNK_LOOK_BACK = 1
VAD_THRESHOLD = 1500  # Voice Activity Detection threshold
TIMEOUT = 600 # in sec

# Whisper use different language codes
WHISPER_ASR_MODEL = settings.get_config("whisper_model")
WHISPER_LANGUAGE_CODES = settings.get_config("whisper_language_codes", 'zh')

GOOGLE_PROJ_ID = settings.get_env("GOOGLE_PROJ_ID")

SYSTEM_PROMPT = settings.get_config("llm_system_prompt", None)
LLM_API_KEY = settings.get_env("BAIDU_AISTUDIO_API_KEY")
LLM_BASE_URL = settings.get_env("BAIDU_AISTUDIO_BASE_URL")
LLM_MODEL = settings.get_config("llm_model")
MAX_TOKENS = 512

GOOGLE_TTS_VOICE = settings.get_config("google_tts_voice", None)
GOOGLE_TTS_LANGUAGE = settings.get_config("google_tts_language", 'cmn-CN')

AUDIO_SAMPLE_RATE = 24000
AUDIO_CHUNK_SIZE = int(AUDIO_SAMPLE_RATE * TIME_PER_CHUNK / 1000)
SAMPLES_PER_FRAME = int(AUDIO_SAMPLE_RATE * TIME_PER_CHUNK / 1000)

AZURE_TTS_KEY = settings.get_env("AZURE_TTS_KEY")
AZURE_TTS_REGION = settings.get_config("azure_tts_region")
# Azure TTS Voice Gallery: [https://speech.microsoft.com/portal/voicegallery]
AZURE_TTS_VOICE = settings.get_config("azure_tts_voice", "zh-CN-YunyangNeural")

ROOT = os.path.dirname(os.path.dirname(__file__))
REEXP = r'(?<=[.!?。，,])\s*'
STOP_WORD = settings.get_config("stop_word", "退出")

