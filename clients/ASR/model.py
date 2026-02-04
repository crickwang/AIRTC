from abc import ABC, abstractmethod
import queue
import sys
from google.cloud import speech_v1 as speech
from funasr import AutoModel
from aiortc import AudioStreamTrack 
import asyncio
import numpy as np
import threading
import traceback
from av.audio.resampler import AudioResampler
from vad.vad import VAD
from register import register
from pywhispercpp.model import Model as whisper_model
from clients.utils import *
from config.constants import TIMEOUT, FUN_ASR_MODEL, STREAMING_LIMIT, ROOT
import os
import uuid
from urllib.parse import urlencode
import requests
import logging

logger = logging.getLogger(__name__)

# ANSI escape sequences for colored output
# These are used to color the output in the terminal.
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
WHITE = "\033[0;37m"

class ASRClient(ABC):
    def __init__(
        self, 
        **kwargs,
    ) -> None:
        super().__init__()
        
    @abstractmethod
    async def generate(
        self: object,
        track: AudioStreamTrack,
        output_queue: asyncio.Queue,
        audio_player: AudioStreamTrack,
        stop_event: asyncio.Event,
        interrupt_event: asyncio.Event,
        **kwargs,
    ):
        """
        Generate a transcription from the audio stream with ASR. \n
        The workflow of this method is to \n
        1. Receive audio frames from the WebRTC track. 
        2. Process the audio frames, generate the transcription, and \
            put it to the output_queue to send them to the next LLM client.
        3. Pass in an audio player to signal the interruption on the WebRTC (front end) \
            track to stop it from generating any audio.
        4. Use the stop_event to signal when to stop the whole program and terminate everything.
        5. Use the interrupt_event to signal when to interrupt other clients (back end). \n
        
        Args:
            track (AudioStreamTrack): Received audio track from WebRTC.
            output_queue (asyncio.Queue): Queue to send transcription results to LLM.
            audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
                                             mainly to enable interruption.
            stop_event (asyncio.Event): Event to signal when to stop connection 
                                        and terminate the program.
            interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
                                            as the user interrupts the current audio playback.
            **kwargs: Additional keyword arguments.
        """
        pass

@register.add_model("asr", "google")
class GoogleASR(ASRClient):
    def __init__(
        self: object, 
        rate: int, 
        language_code: str, 
        chunk_size: int,
        stop_word: str = None,
        **kwargs,
    ) -> None:
        """
        Initializes the Google ASR client.
        Args:
            self (object): The class instance.
            rate (int): The audio file's sampling rate. Defaults to 24000.
            language_code (str): The audio file's language code. Defaults to 'zh-CN'.
                                 please refer to
                                 [https://developers.google.com/workspace/admin/directory/v1/languages]
            chunk_size (int): The audio file's chunk size. Defaults to 240.
            stop_word (str, optional): The word that will trigger the termination of the audio stream.
        """
        super().__init__()
        self.rate = rate
        self.chunk_size = chunk_size
        self.language_code = language_code
        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code=language_code,
            max_alternatives=1,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
            single_utterance=True,
        )
        self.audio_queue = None
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self.is_speaking = False
        self.stop_word = stop_word
        self.pc = kwargs.get('pc', None)
        
    def __enter__(self: object) -> object:
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self.closed = True
        self.audio_queue.put(None)

    def init_queue(self: object, audio_queue: queue.Queue) -> None:
        """
        Initializes the audio queue.

        Args:
            audio_queue (queue.Queue): Queue to hold audio data.
        """
        self.audio_queue = audio_queue
        
    def reset(self: object) -> None:
        """
        Resets the ASR client state.
        Remember to initialize the queue after resetting.
        """
        self.audio_queue = None
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self.is_speaking = False

    def generate_pcm(self: object):
        """
        Generator to yield pcm audio data from the track.
        This method continuously yields audio data chunks from the audio queue.
        Args:
            self (object): The class instance.
        Yields:
            bytes: The audio data chunk.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self.audio_queue.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self.audio_queue.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)

    def listen_print_loop(self: object, responses: object) -> None:
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.
        """
        transcript = ""
        try: 
            if responses is None:
                raise ValueError("No responses to process. Please call get_response() first.")
            
            for response in responses:
                if get_current_time() - self.start_time > STREAMING_LIMIT:
                    self.start_time = get_current_time()
                    break

                if not response.results:
                    continue

                result = response.results[0]

                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                result_seconds = 0
                result_micros = 0

                if result.result_end_time.seconds:
                    result_seconds = result.result_end_time.seconds

                if result.result_end_time.microseconds:
                    result_micros = result.result_end_time.microseconds

                self.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

                corrected_time = (
                    self.result_end_time
                    - self.bridging_offset
                    + (STREAMING_LIMIT * self.restart_counter)
                )
                # Display interim results, but with a carriage return at the end of the
                # line, so subsequent lines will overwrite them.

                if result.is_final:
                    sys.stdout.write(GREEN)
                    sys.stdout.write("\033[K")
                    sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

                    self.is_final_end_time = self.result_end_time
                    self.last_transcript_was_final = True

                    sys.stdout.write(WHITE)
                    sys.stdout.write("\033[K")
                    self.is_speaking = False
                else:
                    sys.stdout.write(RED)
                    sys.stdout.write("\033[K")
                    sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")
                    self.is_speaking = True
                    self.last_transcript_was_final = False
        except Exception as e:
            msg = f"Error in ASR: {e}"
            logger.error(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
        return transcript
    
    def send_request(self):
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in self.generate_pcm()
        )
        responses = self.client.streaming_recognize(
            self.streaming_config, 
            requests,
        )
        transcript = self.listen_print_loop(responses)
        return transcript
    
    async def generate(
        self: object,
        track: AudioStreamTrack, 
        output_queue: asyncio.Queue,
        audio_player: AudioStreamTrack,
        stop_event: asyncio.Event, 
        interrupt_event: asyncio.Event,
        vad: VAD = None,
        timeout: float = TIMEOUT,
        resampler: AudioResampler = None,
        **kwargs,
    ) -> None:
        """
        Google ASR transcription
        Args:
            track (AudioStreamTrack): Received audio track from WebRTC.
            output_queue (asyncio.Queue): Queue to send transcription results to LLM.
            audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
                                             mainly to enable interruption.
            stop_event (asyncio.Event): Event to signal when to stop connection 
                                        and terminate the program.
            interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
                                            as the user interrupts the current audio playback.
            vad (VAD, optional): Voice activity detector instance for detecting speech. This VAD instance
                            can be customized on your own, but it must contain a is_speech(frame) 
                            method that returns a boolean given a np.ndarray frame,
                            distinguishing between speech and silence. If not provided, defaults to None,
                            and every frame will be passed through ASR. Default VAD is None.
            timeout (float, optional): Timeout for ASR. Defaults to 600 seconds.
            resampler (AudioResampler, optional): Audio resampler instance for resampling audio frames.
                                                   If not provided, defaults to None.
        Returns:
            None    
        Raises:
            TimeoutError: If the operation times out.
        """
        try:
            while not stop_event.is_set():
                # Receive frame from WebRTC and process it into numpy array
                frame = await asyncio.wait_for(track.recv(), timeout=timeout)
                frame = resampler.resample(frame)[0].to_ndarray()
                frame = np.squeeze(frame, axis=0).astype(np.int16)
                
                # Check for speech with VAD
                if not vad or vad.is_speech(frame):
                    # New speech detected, interrupt audio playback
                    audio_player.request_interrupt()
                    interrupt_event.set()

                    msg = f"ASR: Speech detected - starting transcription"
                    logger.info(msg)
                    if self.pc:
                        server_to_client(self.pc.log_channel, msg)

                    # queue that contains pcm data, feed into google STT API
                    # wrapped in AudioStream instance
                    audio_queue = queue.Queue()
                    # queue that connects inner STT transcription thread with the main thread
                    # that receives audio from WebRTC
                    session_output_queue = queue.Queue(maxsize=1)
                    self.reset()
                    self.init_queue(audio_queue)
                    # from numpy array to pcm data
                    audio_bytes = frame.tobytes()
                    audio_queue.put(audio_bytes)

                    # Start Google ASR in a separate thread
                    def run_google_stream():
                        """Run Google ASR in separate thread."""
                        try:
                            with self:
                                transcript = self.send_request()
                                if self.stop_word and self.stop_word == transcript:
                                    msg = "Stop word triggered, exiting the program"
                                    logger.info(msg)
                                    if self.pc:
                                        server_to_client(self.pc.log_channel, msg)
                                    stop_event.set()
                                session_output_queue.put(transcript)
                        except Exception as e:
                            msg = f"ASR thread error: {e}"
                            logger.error(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            session_output_queue.put(None)
                    
                    asr_thread = threading.Thread(target=run_google_stream, daemon=True)
                    asr_thread.start()
                    
                    # Continue collecting audio until speech ends
                    while True:
                        try:
                            if stop_event.is_set():
                                self.closed = True
                                await output_queue.put(None)
                                asr_thread.join(timeout=0)
                                break
                            # if current transcription in ended without interruption
                            if self.last_transcript_was_final:
                                self.closed = True
                                asr_thread.join(timeout=5)
                                transcript = session_output_queue.get_nowait() if not session_output_queue.empty() else None
                                await output_queue.put(transcript)
                                if self.pc:
                                    msg = f"ASR: Transcription result - {transcript}"
                                    server_to_client(self.pc.log_channel, msg)
                                interrupt_event.clear()
                                break      
                            # continue to get frame
                            frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                            frame = resampler.resample(frame)[0].to_ndarray()
                            frame = np.squeeze(frame, axis=0).astype(np.int16)
                            audio_bytes = frame.tobytes()
                            audio_queue.put(audio_bytes)
                                
                        except asyncio.TimeoutError:
                            msg = "ASR: Timeout waiting for more audio"
                            logger.info(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            self.closed = True
                            break
        except asyncio.TimeoutError:
            msg = f"ASR: User inactive for {timeout} seconds"
            logger.info(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
        except Exception as e:
            msg = f"ASR: Error: {e}"
            logger.error(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
            traceback.print_exc()
        finally:
            await output_queue.put(None)
     
@register.add_model("asr", "whisper")       
class WhisperASR(ASRClient):
    """
    Whisper ASR model
    """
    def __init__(
        self, 
        model: str = 'tiny', 
        model_dir: str = None, 
        language_code: str = 'zh',
        stop_word: str = None,
        **kwargs):
        """
        Initialize Whisper ASR model.
        Args:
            model (str, optional): Whisper model name. Defaults to tiny.
            model_dir (str, optional): Directory containing model files. Defaults to None.
            language_code (str, optional): Language code for ASR. Defaults to 'zh'.
            stop_word (str, optional): Stop word for ASR. Defaults to None.
        """
        super().__init__()
        self.model = whisper_model(model=model, models_dir=model_dir, **kwargs)
        self.closed = True
        self.audio_queue = None
        self.language_code = language_code
        self.stop_word = stop_word
        self.pc = kwargs.get('pc', None)

    async def generate(
        self: object,
        track: AudioStreamTrack, 
        output_queue: asyncio.Queue,
        audio_player: AudioStreamTrack,
        stop_event: asyncio.Event, 
        interrupt_event: asyncio.Event,
        max_processes: int = 1,
        vad: VAD = None,
        timeout: float = TIMEOUT,
        resampler: AudioResampler = None,
        max_silence_chunk: int = 50,
        **kwargs,
    ) -> None:
        """
        Google ASR transcription
        Args:
            track (AudioStreamTrack): Received audio track from WebRTC.
            output_queue (asyncio.Queue): Queue to send transcription results to LLM.
            audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
                                             mainly to enable interruption.
            stop_event (asyncio.Event): Event to signal when to stop connection 
                                        and terminate the program.
            interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
                                            as the user interrupts the current audio playback.
            max_processes (int): Maximum number of concurrent processes for ASR. Defaults to 1.
            vad (VAD, optional): Voice activity detector instance for detecting speech. This VAD instance
                            can be customized on your own, but it must contain a is_speech(frame) 
                            method that returns a boolean given a np.ndarray frame,
                            distinguishing between speech and silence. If not provided, defaults to None,
                            and every frame will be passed through ASR.
            timeout (float, optional): Timeout for ASR. Defaults to 600 seconds.
            resampler (AudioResampler, optional): Audio resampler instance for resampling audio frames.
        Returns:
            None    
        Raises:
            TimeoutError: If the operation times out.
        """
        try:
            while not stop_event.is_set():
                frame = await asyncio.wait_for(track.recv(), timeout=timeout)
                frame = resampler.resample(frame)[0].to_ndarray()
                frame = np.squeeze(frame, axis=0).astype(np.int16)
                
                # Check for speech with VAD
                if not vad or vad.is_speech(frame):
                    audio_player.request_interrupt()
                    interrupt_event.set()
                    
                    msg = f"ASR: Speech detected - starting transcription"
                    print(msg)
                    if self.pc:
                        server_to_client(self.pc.log_channel, msg)
                    # queue that contains pcm data, feed into google STT API
                    # wrapped in AudioStream instance
                    audio_queue = queue.Queue()
                    # queue that connects inner STT transcription thread with the main thread
                    # that receives audio from WebRTC
                    session_output_queue = queue.Queue(maxsize=1)
                    self.closed = False
                    self.audio_queue = audio_queue  
                    audio_queue.put(frame)
                    
                    def run_whisper_stream():
                        """Run Whisper ASR in separate thread."""
                        try:
                            tot_transcript = ""
                            with self:
                                temp_buffer = []
                                while True:
                                    speech = audio_queue.get(block=True, timeout=1)
                                    temp_buffer.append(speech)
                                    if len(temp_buffer) == max_processes:
                                        transcript = self.model.transcribe(
                                            media=np.array(temp_buffer, dtype=np.int16),
                                            n_processors=max_processes,
                                            language=self.language_code,
                                        )
                                        temp_buffer = []
                                        tot_transcript += transcript[0].text.strip()
                        except queue.Empty:
                            transcript = self.model.transcribe(
                                media=np.array(temp_buffer, dtype=np.int16),
                                n_processors=len(temp_buffer),
                                language=self.language_code,
                            )
                            tot_transcript += transcript[0].text.strip()
                            temp_buffer = []
                            if self.stop_word and self.stop_word == tot_transcript:
                                msg = f"ASR: Stop word '{self.stop_word}' triggered, exiting the program"
                                print(msg)
                                if self.pc:
                                    server_to_client(self.pc.log_channel, msg)
                                stop_event.set()
                            session_output_queue.put(tot_transcript)
                        except Exception as e:
                            msg = f"ASR thread error: {e}"
                            print(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            session_output_queue.put(None)
                    
                    asr_thread = threading.Thread(target=run_whisper_stream, daemon=True)
                    asr_thread.start()
                    
                    # Continue collecting audio until speech ends
                    silence_count = 0
                    while True:
                        try:
                            if not vad.is_speech(frame):
                                silence_count += 1
                            else:
                                silence_count = 0
                            if stop_event.is_set():
                                self.closed = True
                                await output_queue.put(None)
                                asr_thread.join(timeout=0)
                                break
                            if silence_count > max_silence_chunk:
                                self.closed = True
                                asr_thread.join(timeout=10)
                                transcript = session_output_queue.get_nowait() if not session_output_queue.empty() else None
                                await output_queue.put(transcript)
                                interrupt_event.clear()
                                break         
                            frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                            frame = resampler.resample(frame)[0].to_ndarray()
                            frame = np.squeeze(frame, axis=0).astype(np.int16)
                            audio_queue.put(frame)

                        except asyncio.TimeoutError:
                            msg = "ASR: Timeout waiting for more audio"
                            print(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            self.closed = True
                            break
        except asyncio.TimeoutError:
            msg = f"ASR: User inactive for {timeout} seconds"
            print(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
        except Exception as e:
            msg = f"ASR: Error: {e}"
            print(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
            traceback.print_exc()
        finally:
            await output_queue.put(None)

@register.add_model("asr", "funasr")
class FunASR(ASRClient):
    '''Streaming speech recognition using the Paraformer model.'''
    def __init__(
        self, 
        model_path: str = FUN_ASR_MODEL, 
        chunk_size: list = [0, 5, 5], 
        encoder_chunk_look_back: int = 4, 
        decoder_chunk_look_back: int = 1,
        stop_word: str = None,
        **kwargs
    ) -> None:
        '''
        Initialize the ParaformerStreaming model.
        Args:
            model_path (str): Directory to the Pre-trained Paraformer model. Default to FUN_ASR_MODEL
            chunk_size (list): List containing [unknown, frame size in seconds, number of looks back for encoder and decoder].
                               Given a conventional sample rate of 16000Hz and 60ms each frame 
                               (idk why, seems that paraformer just set it that way),
                               the first element is unknown, you may go to github issue at 
                               [https://github.com/modelscope/FunASR/issues/1026] but there is nothing really.
                               The second element is the frame size, and 
                               the third element is the number of looks back for encoder and decoder.
                                1. chunk_size[0]: Unknown, typically set to 0.
                                2. chunk_size[1]: Frame size in seconds (e.g., 10 means 10 * 60ms = 600ms).
                                3. chunk_size[2]: Number of looks back for encoder and decoder (e.g., 5 means looking back 5 frames).
            encoder_chunk_look_back (int): Number of chunks to look back for encoder self-attention.
            decoder_chunk_look_back (int): Number of encoder chunks to look back for decoder cross-attention.
        '''
        self.model = AutoModel(model=os.path.join(ROOT, 'model', model_path))
        self.chunk_size = chunk_size
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.cache = {}
        self.chunk_stride = chunk_size[1] * 960  # Assuming 16kHz sample rate, 60ms frame = 960 samples
        self.stop_word = stop_word
        self.pc = kwargs.get('pc', None)
    
    async def generate(
        self: object,
        track: AudioStreamTrack, 
        output_queue: asyncio.Queue,
        audio_player: AudioStreamTrack,
        stop_event: asyncio.Event, 
        interrupt_event: asyncio.Event,
        vad: VAD = None,
        timeout: float = TIMEOUT,
        resampler: AudioResampler = None,
        max_silence_chunk: int = 50,
        **kwargs,
    ) -> None:
        """
        Google ASR transcription
        Args:
            track (AudioStreamTrack): Received audio track from WebRTC.
            output_queue (asyncio.Queue): Queue to send transcription results to LLM.
            audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
                                             mainly to enable interruption.
            stop_event (asyncio.Event): Event to signal when to stop connection 
                                        and terminate the program.
            interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
                                            as the user interrupts the current audio playback.
            vad (optional): Voice activity detector instance for detecting speech. This VAD instance
                            can be customized on your own, but it must contain a is_speech(frame) 
                            method that returns a boolean given a np.ndarray frame,
                            distinguishing between speech and silence. If not provided, defaults to None,
                            and every frame will be passed through ASR.
            timeout (float, optional): Timeout for ASR. Defaults to 600 seconds
        Returns:
            None    
        Raises:
            TimeoutError: If the operation times out.
        """
        try:
            while not stop_event.is_set():
                frame = await asyncio.wait_for(track.recv(), timeout=timeout)
                frame = resampler.resample(frame)[0].to_ndarray()
                frame = np.squeeze(frame, axis=0).astype(np.int16)
                
                # Check for speech with VAD
                if not vad or vad.is_speech(frame):
                    audio_player.request_interrupt()
                    interrupt_event.set()
                    
                    msg = "ASR: Speech detected - starting transcription"
                    print(msg)
                    if self.pc:
                        server_to_client(self.pc.log_channel, msg)
                    
                    # queue that contains pcm data, feed into google STT API
                    # wrapped in AudioStream instance
                    audio_queue = queue.Queue()
                    # queue that connects inner STT transcription thread with the main thread
                    # that receives audio from WebRTC
                    session_output_queue = queue.Queue(maxsize=1)
                    self.closed = False
                    self.audio_queue = audio_queue  
                    audio_queue.put(frame)
                    
                    def run_paraformer():
                        '''
                        Generate transcription from the input queue.
                        '''
                        try:
                            while True:
                                speech = audio_queue.get(block=True, timeout=1)
                                text = ''
                                total_chunk_num = int(len((speech)-1)/self.chunk_stride+1)
                                for i in range(total_chunk_num):
                                    speech_chunk = speech[i*self.chunk_stride:(i+1)*self.chunk_stride]
                                    is_final = i == total_chunk_num - 1
                                    res = self.model.generate(input=speech_chunk, 
                                                            cache=self.cache, 
                                                            is_final=is_final, 
                                                            chunk_size=self.chunk_size, 
                                                            encoder_chunk_look_back=self.encoder_chunk_look_back, 
                                                            decoder_chunk_look_back=self.decoder_chunk_look_back)
                                    text += res[0]['text'] if res else ''
                                if self.stop_word and self.stop_word == text.strip():
                                    msg = "Stop word triggered, exiting the program"
                                    print(msg)
                                    if self.pc:
                                        server_to_client(self.pc.log_channel, msg)
                                    stop_event.set()
                                session_output_queue.put(text)
                        except queue.Empty:
                            pass
                        except Exception as e:
                            msg = f"ASR: Error in transcription thread: {e}"
                            print(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)

                    asr_thread = threading.Thread(target=run_paraformer, daemon=True)
                    asr_thread.start()
                    
                    # Continue collecting audio until speech ends
                    silence_count = 0
                    while True:
                        try:
                            if not vad.is_speech(frame):
                                silence_count += 1
                            else:
                                silence_count = 0
                            if stop_event.is_set():
                                self.closed = True
                                await output_queue.put(None)
                                asr_thread.join(timeout=0)
                                break
                            if silence_count > max_silence_chunk:
                                self.closed = True
                                asr_thread.join(timeout=10)
                                transcript = session_output_queue.get_nowait() if not session_output_queue.empty() else None
                                await output_queue.put(transcript)
                                if self.pc and transcript:
                                    msg = f"ASR: Transcription result - {transcript}"
                                    server_to_client(self.pc.log_channel, msg)
                                interrupt_event.clear()
                                break         
                            frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                            frame = resampler.resample(frame)[0].to_ndarray()
                            frame = np.squeeze(frame, axis=0).astype(np.int16)
                            audio_queue.put(frame)

                        except asyncio.TimeoutError:
                            msg = "ASR: Timeout waiting for more audio"
                            print(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            self.closed = True
                            break
        except asyncio.TimeoutError:
            msg = f"ASR: User inactive for {timeout} seconds"
            print(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
        except Exception as e:
            msg = f"ASR: Error: {e}"
            print(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
            traceback.print_exc()
        finally:
            await output_queue.put(None)
            
# WebSocket outdated. Not working!!!
        
# This is not streaming ASR!!!
@register.add_model("asr", "baidu")
class BaiduASR(ASRClient):
    def __init__(
        self, 
        access_token: str,
        dev_pid: str,
        uri: str,
        stop_word: str = None,
        **kwargs,
    ) -> None:
        """
        This is not streaming ASR!!!
        Initialize Baidu ASR client.
        Args:
            app_id (str): Baidu app id.
            api_key (str): Baidu api key.
            secret_key (str): Baidu secret key.
            dev_pid (str): Baidu dev pid.
            stop_word (str, optional): Stop word to terminate the program. Defaults to None.
        """
        self.cuid = str(uuid.uuid4())
        self.access_token = access_token
        self.dev_pid = dev_pid
        self.uri = uri
        self.stop_word = stop_word
        self.sample_rate = 16000
        self.format = "pcm"
        self.pc = kwargs.get('pc', None)

    async def generate(
        self, 
        track: AudioStreamTrack, 
        output_queue: asyncio.Queue,
        audio_player: AudioStreamTrack,
        stop_event: asyncio.Event, 
        interrupt_event: asyncio.Event,
        vad: VAD = None,
        timeout: float = TIMEOUT,
        resampler: AudioResampler = None,
        max_silence_chunk: int = 50,
        **kwargs,
    ) -> None:
        """
        Baidu ASR transcription
        Args:
            track (AudioStreamTrack): Received audio track from WebRTC.
            output_queue (asyncio.Queue): Queue to send transcription results to LLM.
            audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
                                             mainly to enable interruption.
            stop_event (asyncio.Event): Event to signal when to stop connection 
                                        and terminate the program.
            interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
                                            as the user interrupts the current audio playback.
            vad (VAD, optional): Voice activity detector instance for detecting speech. This VAD instance
                            can be customized on your own, but it must contain a is_speech(frame) 
                            method that returns a boolean given a np.ndarray frame,
                            distinguishing between speech and silence. If not provided, defaults to None,
                            and every frame will be passed through ASR.
            timeout (float, optional): Timeout for ASR. Defaults to 600 seconds.
            resampler (AudioResampler, optional): Audio resampler instance for resampling audio frames.
                                                   If not provided, defaults to None.
        """
        try:
            while not stop_event.is_set():
                # Receive frame from WebRTC and process it into numpy array
                frame = await asyncio.wait_for(track.recv(), timeout=timeout)
                frame = resampler.resample(frame)[0].to_ndarray()
                frame = np.squeeze(frame, axis=0).astype(np.int16)
                
                # Check for speech with VAD
                if not vad or vad.is_speech(frame):
                    # New speech detected, interrupt audio playback
                    audio_player.request_interrupt()
                    interrupt_event.set()
                    audio = b''

                    msg = f"ASR: Speech detected - starting transcription"
                    print(msg)
                    if self.pc:
                        server_to_client(self.pc.log_channel, msg)


                    # from numpy array to pcm data
                    audio_bytes = frame.tobytes()
                    audio += audio_bytes

                    # Continue collecting audio until speech ends
                    silence_count = 0
                    while True:
                        try:
                            if not vad.is_speech(frame):
                                silence_count += 1
                            else:
                                silence_count = 0
                            if stop_event.is_set():
                                self.closed = True
                                await output_queue.put(None)
                                break
                            # if current transcription in ended without interruption
                            if silence_count > max_silence_chunk:
                                self.closed = True
                                transcript = self.send_request(audio)
                                if transcript is None:
                                    raise ValueError("No transcription result from current ASR session")
                                await output_queue.put(transcript)
                                if self.pc:
                                    msg = f"ASR: Transcription result - {transcript}"
                                    server_to_client(self.pc.log_channel, msg)
                                interrupt_event.clear()
                                break      
                            # continue to get frame
                            frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                            frame = resampler.resample(frame)[0].to_ndarray()
                            frame = np.squeeze(frame, axis=0).astype(np.int16)
                            audio_bytes = frame.tobytes()
                            audio += audio_bytes

                        except ValueError as e:
                            msg = f"ASR: {e}"
                            print(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            break
                        except asyncio.TimeoutError:
                            msg = "ASR: Timeout waiting for more audio"
                            print(msg)
                            if self.pc:
                                server_to_client(self.pc.log_channel, msg)
                            self.closed = True
                            break
        except asyncio.TimeoutError:
            msg = f"ASR: User inactive for {timeout} seconds"
            print(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
        except Exception as e:
            msg = f"ASR: Error: {e}"
            print(msg)
            if self.pc:
                server_to_client(self.pc.log_channel, msg)
            traceback.print_exc()
        finally:
            await output_queue.put(None)
            
    def send_request(self, audio: bytes) -> str:
        """
        Send audio data to the ASR service and return the transcription result.
        Args:
            audio (bytes): The audio data to transcribe.
        Returns:
            str: The transcription result. None if transcription fails.
        """
        params = {
            "cuid": self.cuid,
            "token": self.access_token,
            "dev_pid": self.dev_pid,
        }
        params_query = urlencode(params)
        headers = {
            "Content-Type": "audio/" + self.format + "; rate=" + str(self.sample_rate),
            "Content-Length": str(len(audio))
        }
        url = self.uri + "?" + params_query
        response = requests.post(url, data=audio, headers=headers)
        try:
            assert response.status_code == 200
            return response.json()['result'][0]
        except AssertionError:
            print(f"ASR request failed: {response.status_code}")
        except (ValueError, KeyError) as e:
            print(f"Error parsing ASR response: {e}")
        return None

# @register.add_model("asr", "baiduWS")
# class BaiduWSASR(ASRClient):
#     def __init__(
#         self, 
#         app_id: str, 
#         api_key: str, 
#         dev_pid: str,
#         stop_word: str,
#         uri: str,
#         **kwargs,
#     ) -> None:
#         self.stop_word = stop_word
#         self.app_id = app_id
#         self.api_key = api_key
#         self.dev_pid = dev_pid
#         self.uri = uri
#         self.sample_rate = 16000
#         self.time_per_chunk = 160 / 1000 # in seconds
#         self.format = "pcm"
#         # Nothing in this websocket makes sense. Even WebSocketApp class
#         # does not exist in this package. It's a 6!!! years old documentation.
#         self.ws_app = websocket.WebSocketApp(
#             self.uri,
#             on_open=self.on_open,
#             on_message=self.on_message,
#             on_error=self.on_error,
#             on_close=self.on_close,
#         )
#         self.ws_app.run_forever()

#     async def generate(
#         self, 
#         track: AudioStreamTrack,
#         audio_queue: asyncio.Queue,
#         output_queue: asyncio.Queue,
#         audio_player: AudioStreamTrack,
#         stop_event: asyncio.Event,
#         interrupt_event: asyncio.Event,
#         ws: websocket.WebSocket,
#         vad: VAD = None,
#         timeout: float = TIMEOUT,
#         resampler: AudioResampler = None,
#         **kwargs,
#     ) -> str:
#         """
#         Baidu ASR transcription
#         Args:
#             track (AudioStreamTrack): Received audio track from WebRTC.
#             output_queue (asyncio.Queue): Queue to send transcription results to LLM.
#             audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
#                                              mainly to enable interruption.
#             stop_event (asyncio.Event): Event to signal when to stop connection 
#                                         and terminate the program.
#             interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
#                                             as the user interrupts the current audio playback.
#             vad (VAD, optional): Voice activity detector instance for detecting speech. This VAD instance
#                             can be customized on your own, but it must contain a is_speech(frame) 
#                             method that returns a boolean given a np.ndarray frame,
#                             distinguishing between speech and silence. If not provided, defaults to None,
#                             and every frame will be passed through ASR. Default VAD is None.
#             timeout (float, optional): Timeout for ASR. Defaults to 600 seconds.
#             resampler (AudioResampler, optional): Audio resampler instance for resampling audio frames.
#                                                    If not provided, defaults to None.
#         Returns:
#             None    
#         Raises:
#             TimeoutError: If the operation times out.
#         """

#     async def on_open(
#         self, 
#         track: AudioStreamTrack,
#         audio_queue: asyncio.Queue,
#         output_queue: asyncio.Queue,
#         audio_player: AudioStreamTrack,
#         stop_event: asyncio.Event,
#         interrupt_event: asyncio.Event,
#         ws: websocket.WebSocket,
#         vad: VAD = None,
#         timeout: float = TIMEOUT,
#         resampler: AudioResampler = None,
#         **kwargs,
#     ) -> str:
#         """
#         Baidu ASR transcription
#         Args:
#             track (AudioStreamTrack): Received audio track from WebRTC.
#             output_queue (asyncio.Queue): Queue to send transcription results to LLM.
#             audio_player (AudioStreamTrack): Audio player instance for controlling playback, 
#                                              mainly to enable interruption.
#             stop_event (asyncio.Event): Event to signal when to stop connection 
#                                         and terminate the program.
#             interrupt_event (asyncio.Event): Event to signal when to interrupt other tasks or processes
#                                             as the user interrupts the current audio playback.
#             vad (VAD, optional): Voice activity detector instance for detecting speech. This VAD instance
#                             can be customized on your own, but it must contain a is_speech(frame) 
#                             method that returns a boolean given a np.ndarray frame,
#                             distinguishing between speech and silence. If not provided, defaults to None,
#                             and every frame will be passed through ASR. Default VAD is None.
#             timeout (float, optional): Timeout for ASR. Defaults to 600 seconds.
#             resampler (AudioResampler, optional): Audio resampler instance for resampling audio frames.
#                                                    If not provided, defaults to None.
#         Returns:
#             None    
#         Raises:
#             TimeoutError: If the operation times out.
#         """
#         try:
#             while not stop_event.is_set():
#                 frame = await asyncio.wait_for(track.recv(), timeout=timeout)
#                 self.send_start_params(ws)
#                 frame = resampler.resample(frame)[0].to_ndarray()
#                 frame = np.squeeze(frame, axis=0).astype(np.int16)
                
#                 # Check for speech with VAD
#                 if not vad or vad.is_speech(frame):
#                     audio_player.request_interrupt()
#                     interrupt_event.set()

#                     msg = f"ASR: Speech detected - starting transcription"
#                     print(msg)
#                     if self.pc:
#                         server_to_client(self.pc.log_channel, msg)

#                     # queue that contains pcm data, feed into google STT API
#                     # wrapped in AudioStream instance
#                     audio_queue = queue.Queue()
#                     # queue that connects inner STT transcription thread with the main thread
#                     # that receives audio from WebRTC
#                     session_output_queue = queue.Queue(maxsize=1)
#                     self.reset()
#                     self.init_queue(audio_queue)
#                     audio_bytes = frame.tobytes()
#                     audio_queue.put(audio_bytes)

#                     async def run_baidu_stream():
#                         """Run Baidu ASR in separate thread."""
#                         try:
#                             while True:
#                                 pcm = b''
#                                 while len(pcm) < self.time_per_chunk:
#                                     pcm += audio_queue.get(block=True, timeout=1)
#                                 self.send_audio(ws, pcm)
#                         except queue.Empty:
#                             pass
#                         except Exception as e:
#                             msg = f"ASR thread error: {e}"
#                             print(msg)
#                             if self.pc:
#                                 server_to_client(self.pc.log_channel, msg)

#                     asr_thread = threading.Thread(target=run_baidu_stream, daemon=True)
#                     asr_thread.start()
                    
#                     # Continue collecting audio until speech ends
#                     while True:
#                         silence_count = 0
#                         try:
#                             if not vad.is_speech(frame):
#                                 silence_count += 1
#                             else:
#                                 silence_count = 0
#                             if stop_event.is_set():
#                                 await output_queue.put(None)
#                                 asr_thread.join(timeout=0)
#                                 self.send_cancel(ws)
#                                 break
#                             if silence_count > 100:
#                                 asr_thread.join(timeout=5)
#                                 self.send_finish(ws)
#                                 time.sleep(1)
#                                 transcript = self.on_message(ws, None)
#                                 if self.stop_word and self.stop_word == transcript:
#                                     msg = "Stop word triggered, exiting the program"
#                                     print(msg)
#                                     if self.pc:
#                                         server_to_client(self.pc.log_channel, msg)
#                                     stop_event.set()
#                                 await output_queue.put(transcript)
#                                 if self.pc and transcript:
#                                     msg = f"ASR: Transcription result - {transcript}"
#                                     server_to_client(self.pc.log_channel, msg)
#                                 interrupt_event.clear()
#                                 break         
#                             frame = await asyncio.wait_for(track.recv(), timeout=1.0)
#                             frame = resampler.resample(frame)[0].to_ndarray()
#                             frame = np.squeeze(frame, axis=0).astype(np.int16)
#                             audio_bytes = frame.tobytes()
#                             audio_queue.put(audio_bytes)
                                
#                         except asyncio.TimeoutError:
#                             msg = "ASR: Timeout waiting for more audio"
#                             print(msg)
#                             if self.pc:
#                                 server_to_client(self.pc.log_channel, msg)
#                             self.closed = True
#                             break
#         except asyncio.TimeoutError:
#             msg = f"ASR: User inactive for {timeout} seconds"
#             print(msg)
#             if self.pc:
#                 server_to_client(self.pc.log_channel, msg)
#         except Exception as e:
#             msg = f"ASR: Error: {e}"
#             print(msg)
#             if self.pc:
#                 server_to_client(self.pc.log_channel, msg)
#             traceback.print_exc()
#         finally:
#             await output_queue.put(None)
        

#     def send_start_params(self, ws:websocket.WebSocket):
#         """
#         开始参数帧
#         :param websocket.WebSocket ws:
#         :return:
#         """
#         req = {
#             "type": "START",
#             "data": {
#                 "appid": self.app_id,  # 网页上的appid
#                 "appkey": self.api_key,  # 网页上的appid对应的appkey
#                 "dev_pid": self.dev_pid,  # 识别模型
#                 "cuid": "0",  # 随便填不影响使用。机器的mac或者其它唯一id，百度计算UV用。
#                 "sample": self.sample_rate,  # 固定参数
#                 "format": self.format  # 固定参数
#             }
#         }
#         body = json.dumps(req)
#         ws.send(body, websocket.ABNF.OPCODE_TEXT)

#     def send_audio(self, ws:websocket.WebSocket, pcm):
#         """
#         发送二进制音频数据，注意每个帧之间需要有间隔时间
#         :param  websocket.WebSocket ws:
#         :return:
#         """

#         chunk_len = int(self.sample_rate * 2 / self.time_per_chunk)

#         index = 0
#         total = len(pcm)
#         while index < total:
#             end = index + chunk_len
#             if end >= total:
#                 # 最后一个音频数据帧
#                 end = total
#             body = pcm[index:end]
#             ws.send(body, websocket.ABNF.OPCODE_BINARY)
#             index = end
#             time.sleep(self.time_per_chunk)  # ws.send 也有点耗时，这里没有计算


#     def send_finish(self, ws:websocket.WebSocket):
#         """
#         发送结束帧
#         :param websocket.WebSocket ws:
#         :return:
#         """
#         req = {
#             "type": "FINISH"
#         }
#         body = json.dumps(req)
#         ws.send(body, websocket.ABNF.OPCODE_TEXT)

#     def send_cancel(self, ws:websocket.WebSocket):
#         """
#         发送取消帧
#         :param websocket.WebSocket ws:
#         :return:
#         """
#         req = {
#             "type": "CANCEL"
#         }
#         body = json.dumps(req)
#         ws.send(body, websocket.ABNF.OPCODE_TEXT)

#     def on_open(self, ws:websocket.WebSocket):
#         """
#         连接后发送数据帧
#         :param  websocket.WebSocket ws:
#         :return:
#         """

#         def run(*args):
#             """
#             发送数据帧
#             :param args:
#             :return:
#             """
#             self.send_start_params(ws)
#             self.send_audio(ws)
#             self.send_finish(ws)
#         threading.Thread(target=run).start()


#     def on_message(self, ws:websocket.WebSocket, message):
#         """
#         接收服务端返回的消息
#         :param ws:
#         :param message: json格式，自行解析
#         :return:
#         """
#         print(message)
#         return message


#     def on_error(self, ws:websocket.WebSocket, error):
#         """
#         库的报错，比如连接超时
#         :param ws:
#         :param error: json格式，自行解析
#         :return:
#             """
#         print(str(error))


#     def on_close(self, ws:websocket.WebSocket):
#         """
#         Websocket关闭
#         :param websocket.WebSocket ws:
#         :return:
#         """
#         print("### closed ###")

class ASRClientFactory:
    @staticmethod
    def create(platform, **kwargs) -> ASRClient:
        """
        Create a ASR client instance.
        Args:
            platform (str): The platform name to create the ASR client for.
        Returns:
            ASRClient: The created ASR client.
        """
        try:
            client = register.get_model("asr", platform)
            return client(**kwargs)
        except Exception as e:
            print(f"Error creating ASR client: {e}")
            return None
