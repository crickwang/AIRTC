import asyncio
import json
import logging
import os
import ssl
import traceback
import uuid
import yaml
import argparse
import numpy as np
import threading
import queue
import time
import re
from av import AudioFrame
from pydub import AudioSegment
import io
from fractions import Fraction
import sounddevice as sd
import pyaudio as pa

from dotenv import load_dotenv
from aiohttp import web
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
from av.audio.resampler import AudioResampler
from aiortc.mediastreams import MediaStreamError

# ASR imports
from google.cloud import speech_v1 as speech
from asr.google_transcription import AudioStream, listen_print_loop

# FunASR imports
# from asr.funasr_stream import ParaformerStreaming
# from funasr import AutoModel 

# OpenAI RealTime WebRTC import
# from asr.openai_transcription import OpenAITranscription

# LLM import - OpenAI, Baidu Proxy
from llm.baidu import BaiduClient 

# EdgeTTS import
from tts.edge_tts_stream import EdgeTTS

# Load configuration from YAML file;
# Load environment variables and private tokens from .env file
with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)
load_dotenv()

# Uncomment and set your Google Cloud credentials if needed
# Follow google official documentation to set up your credentials
# [https://cloud.google.com/docs/authentication/getting-started]
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_.json_credential_file"
# os.environ["GOOGLE_CLOUD_PROJECT"] = your_project_id

# Environment Variables and Constants
TIME_PER_CHUNK = 0.96  # in ms
ASR_SAMPLE_RATE = 24000  # in Hz
LANGUAGE_CODE = "zh-CN"  # Language code, please refer to
                         # [https://developers.google.com/workspace/admin/directory/v1/languages]
NUM_SAMPLES_PER_CHUNK = int(ASR_SAMPLE_RATE * TIME_PER_CHUNK)
CHANNELS = 1
FORMAT = "s16"
AUDIO_SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // 10
LAYOUT = "mono"  # Audio layout, mono or stereo
# ASR_MODEL_PATH = config.get("asr_model_path")
# ASR_CHUNK_SIZE = config.get("asr_chunk_size")
# ASR_ENCODER_CHUNK_LOOK_BACK = 4
# ASR_DECODER_CHUNK_LOOK_BACK = 1
# STREAMING_LIMIT = 60000  # 1 minute, in ms
# ENERGY_THRESHOLD = 0.01  # Energy threshold for voice activity detection
CHUNK_SIZE = int(ASR_SAMPLE_RATE / 10)

SYSTEM_PROMPT = config.get("llm_system_prompt", None)
LLM_API_KEY = os.environ.get("BAIDU_AISTUDIO_API_KEY")
LLM_BASE_URL = os.environ.get("BAIDU_AISTUDIO_BASE_URL")
LLM_MODEL = config.get("llm_model")
MAX_TOKENS = 512

ROOT = os.path.dirname(__file__)
REEXP = r'(?<=[.!?。，,])\s*'
logging.basicConfig(filename=os.path.join(ROOT, "a.log"), level=logging.INFO)
logger = logging.getLogger()
pcs = set()

# Model Initialization for FunASR
# asr_llm_queue = asyncio.Queue(maxsize=1)
# asr_model = ParaformerStreaming(
#             model=AutoModel(model=ASR_MODEL_PATH, disable_update=True),
#             chunk_size=ASR_CHUNK_SIZE,
#             encoder_chunk_look_back=ASR_ENCODER_CHUNK_LOOK_BACK,
#             decoder_chunk_look_back=ASR_DECODER_CHUNK_LOOK_BACK
#             )

# Model Initialization for OpenAI RealTime WebRTC
# asr_model = OpenAITranscription()
# client = speech.SpeechClient()
# config = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     sample_rate_hertz=SAMPLE_RATE,
#     language_code=LANGUAGE_CODE,
#     max_alternatives=1,
# )
# streaming_config = speech.StreamingRecognitionConfig(
#     config=config, interim_results=True
# )

# Queues to streamingly store LLM output and TTS audio output
llm_tts_queue = asyncio.Queue()
tts_output_queue = asyncio.Queue()

google_client = speech.SpeechClient()
google_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=ASR_SAMPLE_RATE,
    language_code=LANGUAGE_CODE,
    max_alternatives=1,
)
streaming_config = speech.StreamingRecognitionConfig(
    config=google_config, interim_results=True, single_utterance=True
)

resampler = AudioResampler(rate=AUDIO_SAMPLE_RATE, layout=LAYOUT, format=FORMAT)
llm_client = BaiduClient(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
tts_model = EdgeTTS()

start_time = time.time()

p = pa.PyAudio()
stream = p.open(format=pa.paInt16, channels=1, rate=AUDIO_SAMPLE_RATE, output=True)

class AudioPrinterTrack(AudioStreamTrack):
    """ A track that prints audio data received from the remote peer. """
    kind = "audio"
    def __init__(self: object, track: AudioStreamTrack) -> None:
        """ 
        Initialize the AudioPrinterTrack with the given track. 
        Args:
            self: The class instance.
            track (AudioStreamTrack): The audio track to print data from.
        Returns: 
            None
        """
        super().__init__()
        self.track = track

    async def recv(self):
        """
        Receive an audio frame from the track and print its data.
        Args:
            self: The class instance.
        Returns:
            frame (AudioFrame): The received audio frame.
        """
        frame = await self.track.recv()
        pcm = frame.to_ndarray()
        print(f"Received audio chunk: shape={pcm.shape}, dtype={pcm.dtype} "
              f"with sample rate {frame.sample_rate}")
        return frame

class AudioPlayer(AudioStreamTrack):
    """ A track that plays audio data from a queue. """
    kind = "audio"
    def __init__(self: object, audio_queue: asyncio.Queue, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        """
        Initialize the AudioPlayer with an audio queue and sample rate.
        Args:
            self: The class instance.
            audio_queue (asyncio.Queue): The audio queue to play audio from.
            sample_rate (int, optional): The sample rate of the audio. SAMPLE_RATE Hz by default.
        Returns:
            None
        """
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = sample_rate // 10
        self._audio_buffer = []  # Buffer to store audio data
        self._is_playing = False  # Track if we're actively playing audio
        
    async def recv(self: object) -> AudioFrame:
        """
        Receive an audio frame from the queue and play it.
        Args:
            self: The class instance.
        Returns:
            frame (AudioFrame): The audio frame to be played.
        """
        # Try to fill buffer with available audio data
        while True:
            try:
                audio_data = self.audio_queue.get_nowait()
                
                if audio_data is None:
                    # End of audio stream marker
                    self._is_playing = False
                    break
                else:
                    print('after popping out of the queue', audio_data)
                    #stream.write(audio_data.tobytes())
                    # Convert to list and add to buffer
                    if isinstance(audio_data, np.ndarray):
                        self._audio_buffer.extend(audio_data.tolist())
                    else:
                        self._audio_buffer.extend(audio_data)
                    self._is_playing = True
            except asyncio.QueueEmpty:
                break
        
        # Create frame from buffer or silence
        if len(self._audio_buffer) >= self.samples_per_frame:
            # We have enough audio data
            frame_data = self._audio_buffer[:self.samples_per_frame]
            self._audio_buffer = self._audio_buffer[self.samples_per_frame:]
            audio_array = np.array(frame_data, dtype=np.int16).reshape(1, -1)
        else:
            # Not enough data, create silence but keep consistent timing
            audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
        audio_array = np.array(self._audio_buffer, dtype=np.int16).reshape(1, -1)
        # Create AudioFrame
        stream.write(audio_array.tobytes())
        frame = AudioFrame.from_ndarray(audio_array, format=FORMAT, layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        
        # CRITICAL: Always advance timestamp for consistent timing
        self._timestamp += self.samples_per_frame
        
        #print(f"AudioPlayer frame: pts={frame.pts}, timestamp={self._timestamp}, buffer_size={len(self._audio_buffer)}, playing={self._is_playing}")
        return frame

class TestToneGenerator(AudioStreamTrack):
    """
    Generate a test tone for testing without input audio
    """
    kind = "audio"
    def __init__(self: object, queue: asyncio.Queue, frequency: int = 440, sample_rate: int = 16000) -> None:
        """
        Create a test case for WebRTC Client to receive audio with 
        a queue, frequency, and sample rate.
        Args:
            queue (asyncio.Queue): Queue to put generated audio frames.
            frequency (int, optional): Frequency of the sine wave in Hz. Defaults to 440.
            sample_rate (int, optional): Sample rate of the audio in Hz. Defaults to 16000.
        Returns:
            None
        """
        super().__init__()
        
        self.queue = queue
        self.frequency = frequency
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = SAMPLES_PER_FRAME

    async def recv(self: object) -> AudioFrame:
        """
        Receive an audio frame from the queue, play the sine wave, and return the frame.
        Test how WebRTC Client behave when recieves audio.
        Args:
            self: The class instance.
        Returns:
            frame (AudioFrame): The audio frame to be played.
        Raises:
            MediaStreamError: If there is no more audio data.
        """
        try:
            # Do NOT wait for the queue to be empty, just get the next item
            text = self.queue.get_nowait()
            duration = self.samples_per_frame / self.sample_rate
            t = np.linspace(0, duration, self.samples_per_frame, False)
            
            # Create a sine wave that changes frequency over time for interest
            freq_mod = self.frequency + 50 * np.sin(self._timestamp / self.sample_rate * 0.5)
            sine_wave = np.sin(2 * np.pi * freq_mod * t)
            # Convert to 16-bit PCM
            pcm_data = (sine_wave * 16383).astype(np.int16)  # 50% volume
            audio_array = pcm_data.reshape(1, -1)
            
            # Create AudioFrame
            frame = AudioFrame.from_ndarray(audio_array, format=FORMAT, layout=LAYOUT)
            frame.sample_rate = self.sample_rate
            frame.time_base = Fraction(1, self.sample_rate)
            frame.pts = self._timestamp
            self._timestamp += self.samples_per_frame
            print(type(frame), frame.sample_rate, frame.time_base, frame.pts, frame)
            return frame
        except asyncio.QueueEmpty:
            # If queue is empty, return a silence frame
            return self._create_silence_frame()
    
    def _create_silence_frame(self: object) -> AudioFrame:
        """
        Create a frame of silence.
        Args:
            self: The class instance.
        Returns:
            frame (AudioFrame): A frame of silence.
        """
        silence = np.zeros((1, self.samples_per_frame), dtype=np.int16)
        frame = AudioFrame.from_ndarray(silence, format=FORMAT, layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
        return frame

async def index(request: web.Request) -> web.Response:
    """
    Serve the index HTML file.
    Args:
        request: The HTTP request object.
    Returns:
        web.Response: The response containing the HTML content.
    """
    content = open(os.path.join(ROOT, "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request: web.Request) -> web.Response:
    """
    Serve the JavaScript client file.
    Args:
        request: The HTTP request object.
    Returns:
        web.Response: The response containing the JavaScript content.
    """
    content = open(os.path.join(ROOT, "static/js/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def offer(request: web.Request) -> web.Response:
    """
    Handle the WebRTC offer from the client.
    Args:
        request: The HTTP request object.
    Returns:
        web.Response: The response containing the SDP answer.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # Add generated audio track to the peer connection
    recorder = MediaBlackhole()
    audio_player = AudioPlayer(tts_output_queue)
    pc.addTrack(audio_player)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        """
        Handle connection state changes.
        """
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState in ['closed', 'failed', 'disconnected']:
            # wait for all tasks to finish before closing the connection
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
    async def on_track(track):
        """
        Handle incoming media tracks.
        Args:
            track (AudioStreamTrack): The incoming media track.
        """
        log_info("Track %s received", track.kind)
        if track.kind == "audio":
            # Use the stop event to signal when to stop recording and ASR transcription.
            stop_event = asyncio.Event()
            # Wait for the full transcription to complete
            text = await google_asr(track, stop_event)
            print(f'Transcription result: {text}')
            # Ascynchronously start the LLM generation and TTS transcription,
            # communicate with the respective queues
            pc._llm_task = asyncio.create_task(llm_generator(text, llm_tts_queue))
            pc._tts_task = asyncio.create_task(tts_transcription(llm_tts_queue, tts_output_queue))
            pc._asr_stop_event = stop_event
            # Devour the track
            recorder.addTrack(track)

        @track.on("ended")
        async def on_ended():
            """
            Handle the end of the track.
            """
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

async def on_shutdown(app: web.Application) -> None:
    """
    Handle the shutdown of the application.
    Args:
        app: The web application instance.
    """
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def asr_transcription(track: AudioStreamTrack, stop_event: asyncio.Event = None) -> str:
    '''
    Perform FunASR transcription on the audio track.
    FunASR is an open source streaming ASR model.
    Abandoned due to its poor performance.
    Args:
        track (AudioStreamTrack): The audio track to transcribe.
        stop_event (asyncio.Event, optional): Event to signal when to stop transcription.
            Defaults to None.
    Returns:
        str: Transcription result.
    '''
    text = ''
    frames = None
    try:
        while True:
            if stop_event and stop_event.is_set():
                break
            
            if frames is not None and frames.shape[0] >= NUM_SAMPLES_PER_CHUNK:
                print(f"Processing {frames.shape} samples for ASR transcription")
                # uncomment the asr model first
                transcription = asr_model.generate(frames)
                frames = None
                print(f"Current chunk transcription: {transcription}")
                if transcription and transcription != '':
                    text += transcription
                    
            # Racing
            recv_task = asyncio.create_task(track.recv())
            stop_task = asyncio.create_task(stop_event.wait())
            done, pending = await asyncio.wait([recv_task, stop_task], 
                                               return_when=asyncio.FIRST_COMPLETED)

            if recv_task in done:
                try:
                    frame = recv_task.result()
                    frame = resampler.resample(frame)[0].to_ndarray()
                    frame = np.squeeze(frame, axis=0)
                    frames = np.concatenate((frames, frame), axis=0) if frames is not None else frame
                except MediaStreamError:
                    logger.info(f"User Stops Recording")
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

async def google_asr(track: AudioStreamTrack, stop_event: asyncio.Event = None) -> str:
    """
    Perform ASR transcription on the audio track.
    Args:
        track (AudioStreamTrack): The audio track to transcribe.
        stop_event (asyncio.Event): Event to signal when to stop transcription.
    Returns:
        str: Transcription result.
    Raises:
        Exception: If an error occurs during transcription.
        MediaStreamError: If the media stream is interrupted by the user.
    """
    # pass the incoming track into the AudioStream
    audio_queue = queue.Queue()
    # retrieve the transcription result from the asr_thread
    output_queue = queue.Queue()
    audio_stream = AudioStream(
        audio_queue, 
        rate=ASR_SAMPLE_RATE, 
        chunk_size=CHUNK_SIZE, 
        language_code=LANGUAGE_CODE,
    )
    try:
        def run_google_stream():
            """
            Run the Google ASR streaming transcription in a separate thread.
            """
            with audio_stream:
                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_stream.generate()
                )
                responses = google_client.streaming_recognize(streaming_config, requests)
                transcript = listen_print_loop(responses, audio_stream)
            output_queue.put(transcript)
        asr_thread = threading.Thread(target=run_google_stream, daemon=True)
        frame = await track.recv()
        # Once connection established and recieves the first dummy frame, start the ASR thread
        print("Start recording audio")
        asr_thread.start()
        while True:
            if (stop_event and stop_event.is_set()) or audio_stream.last_transcript_was_final:
                audio_stream.closed = True
                break
            # Continuously receive audio frames from the track
            frame = await track.recv()
            frame = resampler.resample(frame)[0].to_ndarray()
            frame = np.squeeze(frame, axis=0).astype(np.int16)
            audio_bytes = frame.tobytes()
            audio_queue.put(audio_bytes)
    except MediaStreamError:
        logger.info("User stopped recording")      
    except Exception as e:
        logger.error(f"Error during ASR transcription: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # Signal the end of audio stream
        audio_queue.put(None)
    return output_queue.get()

async def llm_generator(text: str, output_queue: asyncio.Queue) -> str:
    '''
    Generate a response from the LLM using the provided prompt.
    Args:
        text (str): The input text to be processed by the LLM.
        output_queue (asyncio.Queue): Queue to put the generated response.
    Returns:
        str: The generated response from the LLM.
    Raises:
        Exception: If an error occurs during LLM generation.
    '''
    output = ''
    try:
        chat_completion = llm_client.generate(
            users=[text],
            system=SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
        )
        for chunk in chat_completion:
            response = chunk.choices[0].delta.content
            if response:
                await output_queue.put(response)
                output += response
        print(f"LLM response: {output}")
        # Indicate end of response  
        await output_queue.put(None)
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")

async def tts_transcription(input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    '''
    Generate TTS audio from the input text.
    Args:
        text (str): Input text to be converted to speech.
        queue (asyncio.Queue): Queue to put audio data.
    Returns:
        None
    '''
    while True:
        text = await input_queue.get()
        if text is None:
            await output_queue.put(None)  # Indicate end of TTS generation
            break
        chunks = re.split(REEXP, text)
        print(chunks)
        for chunk in chunks:
            if not chunk.strip():
                continue
            audio_chunks = await tts_model.process_audio(chunk)
            mp3 = b''.join(audio_chunks)
            audio = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
            audio_np = np.frombuffer(audio.raw_data, dtype=np.int16)
            # print(np.sum(audio_np[audio_np > 0]))
            # print(np.sum(audio_np[audio_np < 0]))
            print(audio.frame_rate)
            #sd.play(audio_np, samplerate=audio.frame_rate)  # Play audio
            # split audio samples into frames of SAMPLES_PER_FRAME
            #p = pa.PyAudio()
            #stream = p.open(format=pa.paInt16, channels=1, rate=24000, output=True)
            for i in range(0, len(audio_np), SAMPLES_PER_FRAME):
                if i + SAMPLES_PER_FRAME <= len(audio_np):
                    audio_frame = audio_np[i:i + SAMPLES_PER_FRAME]
                else:
                    audio_frame = np.pad(audio_np[i:], (0, SAMPLES_PER_FRAME - (len(audio_np) - i)), 'constant')
                if len(audio_frame) != SAMPLES_PER_FRAME:
                    audio_frame = np.pad(audio_frame, (0, SAMPLES_PER_FRAME - len(audio_frame)), 'constant')
                print('before entering the queue', audio_frame)
                await output_queue.put(audio_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal WebRTC audio logger")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/static/js/client.js", javascript)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static/", path=os.path.join(ROOT, "static"), show_index=True)
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)
