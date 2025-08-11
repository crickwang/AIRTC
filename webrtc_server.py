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
SAMPLES_PER_FRAME = AUDIO_SAMPLE_RATE // 100
LAYOUT = "mono"  # Audio layout, mono or stereo
# ASR_MODEL_PATH = config.get("asr_model_path")
# ASR_CHUNK_SIZE = config.get("asr_chunk_size")
# ASR_ENCODER_CHUNK_LOOK_BACK = 4
# ASR_DECODER_CHUNK_LOOK_BACK = 1
# STREAMING_LIMIT = 60000  # 1 minute, in ms
# ENERGY_THRESHOLD = 0.01  # Energy threshold for voice activity detection
CHUNK_SIZE = int(ASR_SAMPLE_RATE / 100)

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

import asyncio
import numpy as np
from fractions import Fraction
from av import AudioFrame
from aiortc import AudioStreamTrack
from pydub import AudioSegment
import io
import re
from collections import deque

# Constants - Use larger chunks for efficiency
AUDIO_SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 240  # 10ms frames for WebRTC
CHUNK_SIZE = 2400  # 100ms chunks for TTS processing

class AudioPlayer(AudioStreamTrack):
    """
    Optimized audio player with better buffering and cleaner shutdown.
    """
    kind = "audio"
    
    def __init__(self, audio_queue: asyncio.Queue, sample_rate: int = AUDIO_SAMPLE_RATE):
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = SAMPLES_PER_FRAME
        
        # Use deque for efficient buffer management
        self._audio_buffer = deque()
        
        # Buffer thresholds
        self._min_buffer = self.samples_per_frame * 20  # 200ms minimum buffer
        self._target_buffer = self.samples_per_frame * 50  # 500ms target buffer
        
        # State tracking
        self._started = False
        self._ending = False
        self._ended = False
        self._frames_sent = 0
        self._total_samples = 0
        self._consecutive_silence = 0
        
        print(f"OptimizedAudioPlayer: Initialized with {self.samples_per_frame} samples/frame")
        
    async def recv(self) -> AudioFrame:
        """
        Generate audio frames with optimized buffering.
        """
        # If we've ended and buffer is empty, just send silence
        if self._ended and len(self._audio_buffer) == 0:
            return self._create_silence_frame()
        
        # Fill buffer if below minimum threshold
        while len(self._audio_buffer) < self._min_buffer and not self._ending:
            try:
                # Longer timeout when waiting for initial data
                timeout = 0.1 if not self._started else 0.01
                
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=timeout
                )
                
                if audio_data is None:
                    # End marker received
                    self._ending = True
                    print(f"AudioPlayer: End marker received, buffer has {len(self._audio_buffer)} samples")
                    break
                    
                # Add to buffer
                if isinstance(audio_data, np.ndarray):
                    self._audio_buffer.extend(audio_data.tolist())
                else:
                    self._audio_buffer.extend(audio_data)
                
                self._total_samples += len(audio_data)
                
                if not self._started and len(self._audio_buffer) >= self._target_buffer:
                    self._started = True
                    print(f"AudioPlayer: Starting playback with {len(self._audio_buffer)} samples buffered")
                    
            except asyncio.TimeoutError:
                # If we haven't started yet, keep waiting
                if not self._started and not self._ending:
                    continue
                break
        
        # Generate output frame
        if len(self._audio_buffer) >= self.samples_per_frame:
            # Extract exactly samples_per_frame from buffer
            frame_data = [self._audio_buffer.popleft() for _ in range(self.samples_per_frame)]
            audio_array = np.array(frame_data, dtype=np.int16).reshape(1, -1)
            self._consecutive_silence = 0
            
            # Log periodically
            if self._frames_sent % 100 == 0:
                print(f"AudioPlayer: Frame {self._frames_sent}, buffer={len(self._audio_buffer)}")
                
        else:
            # Buffer underrun - send silence
            audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
            self._consecutive_silence += 1
            
            # If we're ending and buffer is empty, mark as ended
            if self._ending and len(self._audio_buffer) == 0:
                if not self._ended:
                    self._ended = True
                    print(f"AudioPlayer: Playback complete. Total: {self._frames_sent} frames, {self._total_samples} samples")
            
            # Only log first silence or every 100th
            if self._consecutive_silence == 1:
                print(f"AudioPlayer: Buffer underrun at frame {self._frames_sent}")
            elif self._consecutive_silence % 100 == 0:
                print(f"AudioPlayer: {self._consecutive_silence} consecutive silence frames")
        
        # Create frame
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        
        self._timestamp += self.samples_per_frame
        self._frames_sent += 1
        
        return frame
    
    def _create_silence_frame(self):
        """Create a silence frame."""
        audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
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
    
async def debug_queues():
    """Enhanced queue monitoring"""
    while True:
        await asyncio.sleep(1)
        
        # Check if queues exist and are the right objects
        print(f"Queue sizes - LLM->TTS: {llm_tts_queue.qsize()}, "
              f"TTS->Output: {tts_output_queue.qsize()}")
        print(f"Queue IDs - LLM->TTS: {id(llm_tts_queue)}, "
              f"TTS->Output: {id(tts_output_queue)}")
        
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
    Optimized offer handler with better resource management.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PC-{uuid.uuid4().hex[:8]}"
    pcs.add(pc)

    print(f"\n{'='*50}")
    print(f"{pc_id}: New connection from {request.remote}")
    print(f"{'='*50}")

    # Create session-specific queues
    llm_queue = asyncio.Queue(maxsize=50)
    audio_queue = asyncio.Queue(maxsize=500)
    
    # Create and add audio player
    recorder = MediaBlackhole()
    audio_player = AudioPlayer(audio_queue)
    pc.addTrack(audio_player)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"{pc_id}: Connection state: {pc.connectionState}")
        
        if pc.connectionState in ['closed', 'failed', 'disconnected']:
            # Cancel all tasks
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
            print(f"{pc_id}: Connection closed and cleaned up")

    @pc.on("track")
    async def on_track(track):
        print(f"{pc_id}: Received {track.kind} track")
        
        if track.kind == "audio":
            recorder.addTrack(track)
            
            # Create stop event
            stop_event = asyncio.Event()
            pc._stop_event = stop_event
            
            try:
                # Run ASR
                print(f"{pc_id}: Starting ASR")
                text = await google_asr(track, stop_event)
                
                if not text or text.strip() == "":
                    print(f"{pc_id}: Empty ASR result")
                    await audio_queue.put(None)
                    return
                
                print(f"{pc_id}: ASR result: '{text}'")
                
                # Start pipeline tasks
                pc._llm_task = asyncio.create_task(
                    llm_generator(text, llm_queue, llm_client, SYSTEM_PROMPT, MAX_TOKENS)
                )
                pc._tts_task = asyncio.create_task(
                    tts_transcription(llm_queue, audio_queue, tts_model)
                )
                
                print(f"{pc_id}: Pipeline started")
                
            except Exception as e:
                print(f"{pc_id}: Pipeline error: {e}")
                import traceback
                traceback.print_exc()
                await llm_queue.put(None)
                await audio_queue.put(None)

        @track.on("ended")
        async def on_ended():
            print(f"{pc_id}: Track ended")
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

async def llm_generator(text: str, output_queue: asyncio.Queue, llm_client, system_prompt=None, max_tokens=512):
    """
    Optimized LLM generator with better streaming.
    """
    print(f"LLM: Generating response for: '{text}'")
    total_output = ""
    
    try:
        # Stream the response
        chat_completion = llm_client.generate(
            users=[text],
            system=system_prompt,
            max_tokens=max_tokens,
        )
        
        # Buffer for combining small chunks
        buffer = ""
        
        for chunk in chat_completion:
            response = chunk.choices[0].delta.content
            if response:
                buffer += response
                total_output += response
                
                # Send when we have a complete sentence or substantial text
                if any(buffer.endswith(p) for p in ['.', '!', '?', '。', '！', '？', '\n']) or len(buffer) > 50:
                    await output_queue.put(buffer)
                    buffer = ""
        
        # Send any remaining buffer
        if buffer:
            await output_queue.put(buffer)
        
        print(f"LLM: Generated {len(total_output)} characters")
        
    except Exception as e:
        print(f"LLM: Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always send end marker
        await output_queue.put(None)
        print("LLM: Sent end marker")


async def tts_transcription(input_queue: asyncio.Queue, output_queue: asyncio.Queue, tts_model):
    """
    Optimized TTS with larger chunks and better flow control.
    """
    print("TTS: Starting optimized transcription")
    total_samples_generated = 0
    
    try:
        text_buffer = ""
        
        while True:
            try:
                # Get text from LLM
                text = await asyncio.wait_for(input_queue.get(), timeout=30.0)
                
                if text is None:
                    print("TTS: End of input")
                    # Process any remaining buffered text
                    if text_buffer.strip():
                        await process_text_chunk(text_buffer, output_queue, tts_model)
                    break
                
                # Add to buffer
                text_buffer += text
                
                # Process complete sentences
                sentences = re.split(r'(?<=[.!?。！？])\s*', text_buffer)
                
                # Keep the last incomplete sentence in buffer
                if not text_buffer.endswith(('.', '!', '?', '。', '！', '？')):
                    text_buffer = sentences[-1] if sentences else ""
                    sentences = sentences[:-1]
                else:
                    text_buffer = ""
                
                # Process complete sentences
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    samples = await process_text_chunk(sentence, output_queue, tts_model)
                    total_samples_generated += samples
                    
            except asyncio.TimeoutError:
                print("TTS: Timeout waiting for input")
                break
            except Exception as e:
                print(f"TTS: Error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"TTS: Generated {total_samples_generated} total samples")
        
    finally:
        # Wait a bit for audio to be consumed
        await asyncio.sleep(0.5)
        # Send end marker
        await output_queue.put(None)
        print("TTS: Sent end marker")
        
async def process_text_chunk(text: str, output_queue: asyncio.Queue, tts_model) -> int:
    """
    Process a single text chunk and return number of samples generated.
    """
    if not text.strip():
        return 0
    
    print(f"TTS: Processing: '{text[:50]}...'")
    
    try:
        # Generate TTS audio
        audio_chunks = await tts_model.process_audio(text)
        if not audio_chunks:
            print(f"TTS: No audio generated for: {text}")
            return 0
        
        mp3_data = b''.join(audio_chunks)
        audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
        
        # Ensure correct sample rate
        if audio.frame_rate != AUDIO_SAMPLE_RATE:
            audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE)
        
        audio_np = np.frombuffer(audio.raw_data, dtype=np.int16)
        
        # Verify audio is not silent
        if np.all(audio_np == 0):
            print(f"TTS: WARNING - Generated audio is all zeros for: {text}")
            return 0
        
        print(f"TTS: Generated {len(audio_np)} samples")
        
        # Send in larger chunks for efficiency
        chunk_size = CHUNK_SIZE  # 2400 samples = 100ms
        for i in range(0, len(audio_np), chunk_size):
            chunk = audio_np[i:min(i + chunk_size, len(audio_np))]
            
            # Only pad if it's the last chunk and very small
            if len(chunk) < SAMPLES_PER_FRAME:
                chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)), 'constant')
            
            await output_queue.put(chunk)
            
            # Small delay every few chunks to prevent overwhelming
            if (i // chunk_size) % 10 == 0 and i > 0:
                await asyncio.sleep(0.01)
        
        return len(audio_np)
        
    except Exception as e:
        print(f"TTS: Error processing chunk '{text[:30]}...': {e}")
        return 0
        
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
