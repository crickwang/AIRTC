import argparse
import ssl
from clients.ASR.model import ASRClientFactory
from clients.LLM.model import LLMClientFactory
from clients.TTS.model import TTSClientFactory
from vad.vad import VADFactory

def generate_args():
    parser = argparse.ArgumentParser(description="Minimal WebRTC audio logger")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--asr", type=str, default="google", help="ASR model to use")
    parser.add_argument("--llm", type=str, default="baidu", help="LLM model to use")
    parser.add_argument("--tts", type=str, default="azure", help="TTS model to use")
    parser.add_argument("--vad", type=str, default="simple", help="VAD model to use")
    return parser.parse_args()

def get_ssl_context(args):
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
        return ssl_context
    return None

def create_client(model_type, platform, **kwargs):
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
