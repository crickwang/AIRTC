
from utils import *
args = generate_args()

import asyncio
import json
import logging
import os
import traceback
import uuid
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av.audio.resampler import AudioResampler
from config.constants import *
from audio_player.audio_player import AudioPlayer

logging.basicConfig(filename=os.path.join(ROOT, "a.log"), level=logging.INFO)
logger = logging.getLogger()
pcs = set()

async def index(request: web.Request) -> web.Response:
    """
    Serve the index HTML file.
    Args:
        request: The HTTP request object.
    Returns:
        web.Response: The response containing the HTML content.
    """
    content = open(os.path.join(ROOT, "webpage/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request: web.Request) -> web.Response:
    """
    Serve the JavaScript client file.
    Args:
        request: The HTTP request object.
    Returns:
        web.Response: The response containing the JavaScript content.
    """
    content = open(os.path.join(ROOT, "webpage/js/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request: web.Request) -> web.Response:
    """
    Manage WebRTC connection.
    """
    params = await request.json() 
    # located in js
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PC-{uuid.uuid4().hex[:8]}"
    pcs.add(pc)

    print(f"Current {pc_id}: New connection established")

    # Create session-specific queues
    # asr_queue: transfer results of ASR to LLM
    # llm_queue: transfer results of LLM to TTS
    # audio_queue: transfer audio data from TTS to WebRTC browser player
    asr_queue = asyncio.Queue(maxsize=20)
    llm_queue = asyncio.Queue(maxsize=20)
    audio_queue = asyncio.Queue(maxsize=100)
    
    # Create components
    stop_event = asyncio.Event()
    interrupt_event = asyncio.Event()
    recorder = MediaBlackhole()
    audio_player = AudioPlayer(
        audio_queue, 
        sample_rate=AUDIO_SAMPLE_RATE, 
        samples_per_frame=SAMPLES_PER_FRAME, 
        format=FORMAT, 
        layout=LAYOUT,
    )
    vad = create_client("vad", 
                        args.vad, 
                        threshold=VAD_THRESHOLD,
                        )
    resampler = AudioResampler(rate=ASR_SAMPLE_RATE, 
                               layout=LAYOUT, 
                               format=FORMAT,
                               )
    asr_client = create_client("asr", 
                               platform=args.asr, 
                               rate=ASR_SAMPLE_RATE, 
                               language_code=ASR_LANGUAGE, 
                               chunk_size=ASR_CHUNK_SIZE,
                               stop_word=STOP_WORD,
                               )
    llm_client = create_client("llm", 
                               platform=args.llm, 
                               model=LLM_MODEL, 
                               api_key=LLM_API_KEY, 
                               base_url=LLM_BASE_URL,
                               )
    tts_client = create_client("tts", 
                               platform=args.tts, 
                               voice=AZURE_TTS_VOICE, 
                               key=AZURE_TTS_KEY, 
                               region=AZURE_TTS_REGION,
                               )

    pc.addTrack(audio_player)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Current {pc_id}: Connection state: {pc.connectionState}")
        if pc.connectionState == 'connected':
            print('=' * 50)
            print("Start Recording")
            print('=' * 50)
        if pc.connectionState in ['closed', 'failed', 'disconnected']:
            # Cancel all tasks
            if hasattr(pc, '_stop_event'):
                pc._stop_event.set()
            
            tasks = []
            for attr in ['_asr_task', '_llm_task', '_tts_task']:
                if hasattr(pc, attr):
                    task = getattr(pc, attr)
                    if not task.done():
                        task.cancel()
                        tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            await pc.close()
            pcs.discard(pc)
            print(f"{pc_id}: Connection cleaned up")

    @pc.on("track")
    async def on_track(track):
        print(f"Current {pc_id}: Received {track.kind} track")
        if track.kind == "audio":
            recorder.addTrack(track)
            # Create control events to stop or interrupt connection
            pc._stop_event = stop_event
            pc._interrupt_event = interrupt_event
            
            try:
                pc._asr_task = asyncio.create_task(
                    asr_client.generate(
                        track, 
                        asr_queue, 
                        audio_player, 
                        stop_event, 
                        interrupt_event, 
                        vad=vad,
                        resampler=resampler,
                        timeout=TIMEOUT,
                    )
                )

                pc._llm_task = asyncio.create_task(
                    llm_client.generate(
                        asr_queue, 
                        llm_queue, 
                        stop_event, 
                        interrupt_event, 
                        system=SYSTEM_PROMPT, 
                        max_tokens=MAX_TOKENS, 
                        timeout=TIMEOUT,
                    )
                )
                
                pc._tts_task = asyncio.create_task(
                    tts_client.generate(
                        llm_queue, 
                        audio_queue, 
                        stop_event, 
                        interrupt_event,
                        samples_per_frame=SAMPLES_PER_FRAME,
                        timeout=TIMEOUT,
                    )
                )

            except Exception as e:
                print(f"Current {pc_id}: Pipeline error: {e}")
                traceback.print_exc()

        @track.on("ended")
        async def on_ended():
            print(f"Current {pc_id}: Track ended")
            if hasattr(pc, "_stop_event"):
                pc._stop_event.set()
            await recorder.stop()

    # Complete WebRTC setup
    await pc.setRemoteDescription(offer)
    await recorder.start()
    
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )
    
async def on_shutdown(app: web.Application) -> None:
    """
    Handle the shutdown of the application.
    Args:
        app: The web application instance.
    """
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def run_web_app(args, ssl_context=None):
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get(path=os.path.join("/", "webpage", "js", "client.js"), handler=javascript)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static/", path=os.path.join(ROOT, "webpage"), show_index=True)
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)

if __name__ == "__main__":
    ssl_context = get_ssl_context(args)
    run_web_app(args, ssl_context=ssl_context)