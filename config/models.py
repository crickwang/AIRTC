from config.imports import *
from config.constants import *
class ModelConfig:
    def __init__(self):
        pass

class FunASR:
    def __init__(
        self, 
        model_path = ASR_MODEL_PATH, 
        disable_update = True,
        chunk_size = ASR_CHUNK_SIZE,
        encoder_chunk_look_back = ASR_ENCODER_CHUNK_LOOK_BACK,
        decoder_chunk_look_back = ASR_DECODER_CHUNK_LOOK_BACK,
        max_queue_size = 1
    ) -> None:
        super().__init__()
        self.auto_model = AutoModel(model=model_path, disable_update=disable_update)
        self.model_path = model_path
        self.model = ParaformerStreaming(
            model=self.auto_model,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )
        self.output_queue = asyncio.Queue(maxsize=max_queue_size)
        
class GoogleASR:
    def __init__(
        self,
        sample_rate = ASR_SAMPLE_RATE,
        language = ASR_LANGUAGE,
        max_alternatives = 1,
        interim_results = True,
        single_utterance = True
    ):
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language,
            max_alternatives=max_alternatives
        )
        self.steaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=interim_results,
            single_utterance=single_utterance
        )
        
    def send_request(audio_content):
        speech.StreamingRecognizeRequest(audio_content=audio_content)
        
class BaiduLLM:
    def __init__(
        self,
        model_name = LLM_MODEL,
        api_key = LLM_API_KEY,
        base_url = LLM_BASE_URL
    ) -> None:
        self.model = BaiduClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
        
class GoogleTTS:
    pass


