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
TIME_PER_CHUNK = 0.96  # in ms
ASR_SAMPLE_RATE = 24000  # in Hz
ASR_LANGUAGE = settings.get_config("asr_language", "zh-CN")  # Language code, please refer to
                         # [https://developers.google.com/workspace/admin/directory/v1/languages]
NUM_SAMPLES_PER_CHUNK = int(ASR_SAMPLE_RATE * TIME_PER_CHUNK)
CHANNELS = 1
FORMAT = "s16"
AUDIO_SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // 100
LAYOUT = "mono"  # Audio layout, mono or stereo

# Constants for local ASR 
ASR_MODEL_PATH = settings.get_config("asr_model_path")
ASR_CHUNK_SIZE = settings.get_config("asr_chunk_size")
ASR_ENCODER_CHUNK_LOOK_BACK = 4
ASR_DECODER_CHUNK_LOOK_BACK = 1
STREAMING_LIMIT = 60000  # 1 minute, in ms
ENERGY_THRESHOLD = 0.01  # Energy threshold for voice activity detection
CHUNK_SIZE = int(ASR_SAMPLE_RATE / 10)

SYSTEM_PROMPT = settings.get_config("llm_system_prompt", None)
LLM_API_KEY = settings.get_env("BAIDU_AISTUDIO_API_KEY")
LLM_BASE_URL = settings.get_env("BAIDU_AISTUDIO_BASE_URL")
LLM_MODEL = settings.get_config("llm_model")
MAX_TOKENS = 512

TTS_VOICE = settings.get_config("tts_voice", None)
TTS_LANGUAGE = settings.get_config("tts_language", 'cmn-CN')

AZURE_TTS_KEY = settings.get_env("AZURE_TTS_KEY")
AZURE_TTS_REGION = settings.get_config("azure_tts_region")
# Azure TTS Voice Gallery: [https://speech.microsoft.com/portal/voicegallery]
AZURE_TTS_VOICE = settings.get_config("azure_tts_voice", "zh-CN-YunyangNeural")

ROOT = os.path.dirname(os.path.dirname(__file__))
REEXP = r'(?<=[.!?。，,])\s*'


