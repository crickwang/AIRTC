import asyncio
import json
import logging
import os
import ssl
import uuid
from sklearn import base
import yaml
import argparse
import numpy as np
import av

from dotenv import load_dotenv
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from asr.funasr_stream import ParaformerStreaming
from av.audio.resampler import AudioResampler
from funasr import AutoModel   
from dataclasses import dataclass

from llm.baidu import BaiduClient 

from tts.edge_tts_stream import EdgeTTS

with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)
    
load_dotenv()

API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")

TIME_PER_CHUNK = 0.48  # 960ms per chunk, adjust as needed, It's better to be divisible by 16000 Hz
ASR_SAMPLE_RATE = 16000  # Sample rate for ASR model
NUM_SAMPLES_PER_CHUNK = int(ASR_SAMPLE_RATE * TIME_PER_CHUNK)

ASR_MODEL_PATH = config.get("asr_model_path")
ASR_CHUNK_SIZE = config.get("asr_chunk_size")
ASR_ENCODER_CHUNK_LOOK_BACK = 4
ASR_DECODER_CHUNK_LOOK_BACK = 1

SYSTEM_PROMPT = config.get("llm_system_prompt", None)
LLM_MODEL = config.get("llm_model")
MAX_TOKENS = 512

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("pc")
pcs = set()

asr_llm_queue = asyncio.Queue(maxsize=4)
llm_tts_queue = asyncio.Queue(maxsize=4)
asr_model = AutoModel(model=ASR_MODEL_PATH, disable_update=True)
llm_client = BaiduClient(model=LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)
tts_model = EdgeTTS()

# FunASR does not support punctuation, so it is hard to determine the end of a sentence.
# is_final does nothing here.
@dataclass
class PromptText:
    text: str
    is_final: bool = False

class AudioPrinterTrack(MediaStreamTrack):
    kind = "audio"
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        pcm = frame.to_ndarray()
        print(f"Received audio chunk: shape={pcm.shape}, dtype={pcm.dtype} with sample rate {frame.sample_rate}")
        return frame

async def index(request):
    content = open(os.path.join(ROOT, "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "templates/js/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState in ['closed', 'failed', 'disconnected']:
            # Should llm task be cancelled here?
            if hasattr(pc, "_asr_stop_event"):
                pc._asr_stop_event.set()
            if hasattr(pc, "_asr_task"):
                await pc._asr_task
            if hasattr(pc, "_llm_task"):
                await pc._llm_task
            if hasattr(pc, "_tts_task"):
                await pc._tts_task
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            #print(type(track), track)
            #audio_track = AudioPrinterTrack(track)
            stop_event = asyncio.Event()
            pc._asr_task = asyncio.create_task(asr_transcription(track, asr_llm_queue, stop_event))
            pc._llm_task = asyncio.create_task(llm_generator(asr_llm_queue, llm_tts_queue))
            pc._tts_task = asyncio.create_task(tts_transcription(llm_tts_queue))
            pc._asr_stop_event = stop_event
            recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            if hasattr(pc, "_asr_stop_event"):
                pc._asr_stop_event.set()
            if hasattr(pc, "_asr_task"):
                await pc._asr_task
            await recorder.stop()
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    await recorder.start()

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    
async def asr_transcription(track, queue, stop_event=None) -> str:
    '''
    Perform ASR transcription on the audio track.
    Args:
        track (MediaStreamTrack): The audio track to transcribe.
        queue (asyncio.Queue): Queue to put transcription results.
        stop_event (asyncio.Event): Event to signal when to stop transcription.
    Returns:
        str: Transcription result.
    '''
    text = ''
    resampler = AudioResampler(rate=ASR_SAMPLE_RATE, layout='mono', format="s16")
    frames = None
    paraformer = ParaformerStreaming(
                model=asr_model,
                chunk_size=ASR_CHUNK_SIZE,
                encoder_chunk_look_back=ASR_ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=ASR_DECODER_CHUNK_LOOK_BACK
                )
    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            if frames is not None and frames.shape[0] >= NUM_SAMPLES_PER_CHUNK:
                print(f"Processing {frames.shape} samples for ASR transcription")
                transcription = paraformer.generate(frames)
                frames = None
                print(f"Current chunk transcription: {transcription}")
                if transcription and transcription != '':
                    text += transcription
                    await queue.put(PromptText(text=text, is_final=False))
                    
            # Racing
            recv_task = asyncio.create_task(track.recv())
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait([recv_task, stop_task], return_when=asyncio.FIRST_COMPLETED)

            if recv_task in done:
                frame = recv_task.result()
                frame = resampler.resample(frame)[0].to_ndarray()
                frame = np.squeeze(frame, axis=0)
                frames = np.concatenate((frames, frame), axis=0) if frames is not None else frame
            else:
                recv_task.cancel()
                break
            for task in pending:
                task.cancel()
    except Exception as e:
        logger.error(f"Error during ASR transcription: {e}")
    finally:
        await queue.put(None)
    print(f"Final transcription: {text}")
    return text

async def llm_generator(queue1, queue2) -> str:
    '''
    Generate a response from the LLM using the provided prompt.
    Args:
        queue (asyncio.Queue): Queue containing prompts for the LLM.
    Returns:
        str: The generated response from the LLM.
    '''
    # TODO: Record multiple conversations and send them to the LLM
    try:
        while True:
            prompt = await queue1.get()
            if prompt is None:
                break
            response = await asyncio.to_thread(llm_client.generate, 
                                            users=[prompt.text], 
                                            system=SYSTEM_PROMPT, 
                                            max_tokens=MAX_TOKENS)
            await queue2.put(response)
            print(f"LLM response: {response}")
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        
async def tts_transcription(queue):
    '''
    Generate TTS audio from the input text.
    Args:
        text (str): Input text to be converted to speech.
        queue (asyncio.Queue): Queue to put audio data.
    '''
    while True:
        text = await queue.get()
        if text is None:
            break
        print(f"Generating TTS for: {text}")
        await tts_model.generate(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal WebRTC audio logger")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/templates", index)
    app.router.add_get("/templates/js/client.js", javascript)
    app.router.add_post("/offer", offer)
    app.router.add_static("/templates", ".", show_index=True)
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)