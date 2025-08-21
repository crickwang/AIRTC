from config.imports import *
from config.constants import *

logging.basicConfig(filename=os.path.join(ROOT, "a.log"), level=logging.INFO)
logger = logging.getLogger()
pcs = set()

# Model Initialization for FunASR (local)
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
#     language_code=ASR_LANGUAGE,
#     max_alternatives=1,
# )
# streaming_config = speech.StreamingRecognitionConfig(
#     config=config, interim_results=True
# )

# Google Cloud clients for ASR (online)
google_asr_client = speech.SpeechClient()
google_asr_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=ASR_SAMPLE_RATE,
    language_code=ASR_LANGUAGE,
    max_alternatives=1,
)
google_asr_streaming_config = speech.StreamingRecognitionConfig(
    config=google_asr_config, interim_results=True, single_utterance=True
)

# Resampler to change audio properties if needed
resampler = AudioResampler(rate=ASR_SAMPLE_RATE, layout=LAYOUT, format=FORMAT)
llm_client = BaiduClient(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
# tts_model = EdgeTTS()

# # Google Cloud clients for TTS (online)
# google_tts_client = texttospeech.TextToSpeechClient()
# # See https://cloud.google.com/text-to-speech/docs/voices for all voices.
# google_tts_streaming_config = texttospeech.StreamingSynthesizeConfig(
#     voice=texttospeech.VoiceSelectionParams(
#         name=TTS_VOICE,
#         language_code=TTS_LANGUAGE,
#     )
# )
# # Set the config for your stream. The first request must contain your config, 
# # and then each subsequent request must contain text.
# google_tts_config_request = texttospeech.StreamingSynthesizeRequest(
#     streaming_config=google_tts_streaming_config
# )

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

start_time = time.time()
llm_first_chunk_time = 0
llm_start_time = 0
tts_first_chunk_time = 0
tts_start_time = 0

class MultiFrameVAD:
    """
    Voice Activity Detection with multiple frames. More robust against noise.
    Not advised to use this VAD, as it may swallow first several frames. 
    You may adjust you own VAD if you want.
    """
    def __init__(self, threshold=VAD_THRESHOLD, speech_frames_required=3):
        """
        Initialize MultiFrameVAD instance
        Args:
            self: The instance of the class.
            threshold: The energy threshold for detecting speech.
                       In general, noise < 1000 while speech > 2000.
                       Maybe adjusted based on the noise level of the environment.
            speech_frames_required: The number of consecutive frames required to confirm speech.
        """
        self.threshold = threshold
        self.speech_frames_required = speech_frames_required
        self.speech_frame_count = 0
        self.is_currently_speaking = False
        # store the previous frames and prevent them from being swallowed if they
        # are meaningful speeches.
        self.frames = [None for i in range(speech_frames_required)]
        
    def populate(self, frame:np.ndarray) -> None:
        """
        Populate the VAD frames with the current audio frame.
        Args:
            frame (np.ndarray): The audio frame to populate.
        Returns:
            None
        """
        self.frames.pop(0)
        self.frames.append(frame)
        
    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Detect if frame contains speech with hysteresis to avoid false positives.
        Args:
            frame (np.ndarray): The audio frame to analyze.
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        # Calculate RMS energy
        if len(frame) == 0:
            return False
            
        rms_energy = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        
        # Use hysteresis for more stable detection
        self.populate(frame)
        if rms_energy > self.threshold:
            self.speech_frame_count += 1
            if self.speech_frame_count >= self.speech_frames_required:
                if not self.is_currently_speaking:
                    print(f"VAD: Speech detected (energy: {rms_energy:.1f})")
                self.is_currently_speaking = True
                self.speech_frame_count = 0
        return self.is_currently_speaking

class SimpleVAD:
    """
    A simple Voice Activity Detection (VAD) class 
    that only measures the energy of a single audio frame.
    """
    def __init__(self, threshold=VAD_THRESHOLD):
        """
        Initialize SimpleVAD instance.
        Args:
            self: The instance of the class.
            threshold: The energy threshold for detecting speech.
                       In general, noise < 2000 while speech > 20000.
                       Maybe adjusted based on the noise level of the environment.
        """
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Detect if frame contains speech.
        Args:
            frame (np.ndarray): The audio frame to analyze.
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        if len(frame) == 0:
            return False

        rms_energy = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        return rms_energy > self.threshold

class AudioPrinterTrack(AudioStreamTrack):
    """ 
    A track that prints audio data received from the remote peer. 
    Debug only.
    """
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
    """
    AudioPlayer instance that passes results of TTS (asyncio queue) to the WebRTC peer 
    with proper interruption support and buffer management.
    """
    kind = "audio"
    def __init__(self, audio_queue: asyncio.Queue, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = SAMPLES_PER_FRAME
        
        # buffer management
        self._audio_buffer = deque()
        self._max_buffer_ms = 30000  # Maximum 30000ms of audio buffered
        self._max_buffer_samples = (self._max_buffer_ms * sample_rate) // 1000
        
        # State tracking
        self._is_playing = False
        self._interrupt_requested = False
        self._frames_sent = 0
        
        # Timing control
        self._last_frame_time = time.time()
        self._frame_interval = self.samples_per_frame / self.sample_rate
        
        print(f"AudioPlayer: Initialized with {self._max_buffer_ms}ms max buffer")

    async def recv(self) -> AudioFrame:
        """
        Send audio frames to the WebRTC peer.
        Args:
            self: The class instance.
        Returns:
            AudioFrame: The generated audio frame.
        """
        await self._control_timing()
        
        # Handle interruption immediately
        if self._interrupt_requested:
            self._clear_buffers()
            self._interrupt_requested = False
            self._is_playing = False
            print("AudioPlayer: Interrupt processed, buffers cleared")
        
        # Try to fill buffer if not interrupted
        if not self._interrupt_requested:
            await self._fill_buffer()
        
        # Generate frame
        if len(self._audio_buffer) >= self.samples_per_frame and not self._interrupt_requested:
            # Extract audio samples
            frame_samples = [self._audio_buffer.popleft() for _ in range(self.samples_per_frame)]
            audio_array = np.array(frame_samples, dtype=np.int16).reshape(1, -1)
            self._is_playing = True
        else:
            # Send silence if buffer empty or interrupted
            audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
            if self._is_playing:
                print("AudioPlayer: Buffer empty, sending silence")
                self._is_playing = False
        
        # Create and return frame
        frame = AudioFrame.from_ndarray(audio_array, format=FORMAT, layout=LAYOUT)
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        
        self._timestamp += self.samples_per_frame
        self._frames_sent += 1
        return frame
    
    async def _control_timing(self):
        """
        Control frame timing for consistent 10ms intervals.
        """
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        
        if elapsed < self._frame_interval:
            sleep_time = self._frame_interval - elapsed
            if sleep_time > 0.001:
                await asyncio.sleep(sleep_time)
        
        self._last_frame_time = time.time()
    
    async def _fill_buffer(self):
        """Fill audio buffer without blocking."""
        # Don't overfill buffer
        if len(self._audio_buffer) >= self._max_buffer_samples:
            return
        
        # Try to get audio data (non-blocking)
        try:
            audio_data = self.audio_queue.get_nowait()
            
            if audio_data is None:
                # End marker - let buffer drain naturally
                print("AudioPlayer: End marker received")
                return
            
            # Add to buffer
            if isinstance(audio_data, np.ndarray):
                samples_to_add = audio_data.tolist()
            else:
                samples_to_add = list(audio_data)
            
            # Prevent buffer overflow
            available_space = self._max_buffer_samples - len(self._audio_buffer)
            if available_space > 0:
                actual_samples = samples_to_add[:available_space]
                self._audio_buffer.extend(actual_samples)
                
                if len(samples_to_add) > available_space:
                    dropped = len(samples_to_add) - available_space
                    print(f"AudioPlayer: Dropped {dropped} samples due to buffer overflow")
        except asyncio.QueueEmpty:
            pass
    
    def request_interrupt(self):
        """Request immediate interruption of audio playback."""
        self._interrupt_requested = True
        print(f"AudioPlayer: Interrupt requested")
    
    def _clear_buffers(self):
        """Clear all audio buffers immediately."""
        self._audio_buffer.clear()
        
        # Also clear the queue to prevent old audio from playing
        try:
            while True:
                self.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        print(f"AudioPlayer: Buffers cleared")

class TestToneGenerator(AudioStreamTrack):
    """
    Generate a test tone for testing without input audio
    """
    kind = "audio"
    def __init__(
        self: object, 
        queue: asyncio.Queue, 
        frequency: int = 440, 
        sample_rate: int = 16000
        ) -> None:
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
    Manage WebRTC connection.
    """
    params = await request.json() 
    # located in js
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PC-{uuid.uuid4().hex[:8]}"
    pcs.add(pc)

    print(f"{pc_id}: New connection established")

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
    audio_player = AudioPlayer(audio_queue)
    vad = SimpleVAD()
    
    pc.addTrack(audio_player)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"{pc_id}: Connection state: {pc.connectionState}")
        
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
        print(f"{pc_id}: Received {track.kind} track")
        
        if track.kind == "audio":
            recorder.addTrack(track)
            # Create control events to stop or interrupt connection
            pc._stop_event = stop_event
            pc._interrupt_event = interrupt_event
            
            try:
                pc._asr_task = asyncio.create_task(
                    google_asr(
                        track, 
                        asr_queue, 
                        audio_player, 
                        stop_event, 
                        interrupt_event, 
                        vad=vad,
                        sample_rate=ASR_SAMPLE_RATE,
                        chunk_size=ASR_CHUNK_SIZE,
                        language_code=ASR_LANGUAGE,
                        timeout=TIMEOUT,
                    )
                )

                pc._llm_task = asyncio.create_task(
                    baidu_llm(
                        asr_queue, 
                        llm_queue, 
                        llm_client, 
                        stop_event, 
                        interrupt_event, 
                        system=SYSTEM_PROMPT, 
                        max_tokens=MAX_TOKENS, 
                        timeout=TIMEOUT,
                    )
                )
                
                pc._tts_task = asyncio.create_task(
                    azure_tts(
                        llm_queue, 
                        audio_queue, 
                        stop_event, 
                        interrupt_event,
                        timeout=TIMEOUT,
                    )
                )
                
                print(f"{pc_id}:  pipeline started")
                
            except Exception as e:
                print(f"{pc_id}: Pipeline error: {e}")
                traceback.print_exc()

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

async def google_asr(
    track: AudioStreamTrack, 
    output_queue: asyncio.Queue,
    audio_player: AudioPlayer,
    stop_event: asyncio.Event, 
    interrupt_event: asyncio.Event,
    vad=None,
    sample_rate: int = ASR_SAMPLE_RATE,
    chunk_size: int = ASR_CHUNK_SIZE,
    language_code: str = ASR_LANGUAGE,
    timeout: float = TIMEOUT,
) -> None:
    """
    Google ASR transcription
    Args:
        track (AudioStreamTrack): Received audio track from WebRTC.
        output_queue (asyncio.Queue): Queue to send transcription results to LLM.
        audio_player (AudioPlayer): Audio player instance for controlling playback, 
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
        sample_rate (int, optional): Sample rate for ASR. Defaults to 16000.
        chunk_size (int, optional): Chunk size for audio processing. Defaults to 512.
        language_code (str, optional): Language code for ASR. Defaults to 'zh-CN'.
                                       Please refer to constants.py to check for the available
                                       language codes.
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
                
                print("ASR: Speech detected - starting transcription")
                
                # queue that contains pcm data, feed into google STT API
                # wrapped in AudioStream instance
                audio_queue = queue.Queue()
                
                # queue that connects inner STT transcription thread with the main thread
                # that receives audio from WebRTC
                session_output_queue = queue.Queue(maxsize=1)
                
                # Google API instance
                audio_stream = AudioStream(
                    audio_queue, 
                    rate=sample_rate, 
                    chunk_size=chunk_size,
                )

                audio_bytes = frame.tobytes()
                audio_queue.put(audio_bytes)

                def run_google_stream():
                    """Run Google ASR in separate thread."""
                    try:
                        with audio_stream:
                            requests = (
                                speech.StreamingRecognizeRequest(audio_content=content)
                                for content in audio_stream.generate()
                            )
                            responses = google_asr_client.streaming_recognize(
                                google_asr_streaming_config, 
                                requests,
                            )
                            transcript = listen_print_loop(responses, audio_stream)
                        session_output_queue.put(transcript)
                    except Exception as e:
                        print(f"ASR thread error: {e}")
                        session_output_queue.put(None)
                
                asr_thread = threading.Thread(target=run_google_stream, daemon=True)
                asr_thread.start()
                
                # Continue collecting audio until speech ends
                while True:
                    try:
                        if stop_event.is_set():
                            audio_stream.closed = True
                            await output_queue.put(None)
                            asr_thread.join(timeout=0)
                            break
                        if audio_stream.last_transcript_was_final:
                            audio_stream.closed = True
                            asr_thread.join(timeout=5)
                            transcript = session_output_queue.get_nowait() if not session_output_queue.empty() else None
                            await output_queue.put(transcript)
                            interrupt_event.clear()
                            break         
                        frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                        frame = resampler.resample(frame)[0].to_ndarray()
                        frame = np.squeeze(frame, axis=0).astype(np.int16)
                        audio_bytes = frame.tobytes()
                        audio_queue.put(audio_bytes)
                            
                    except asyncio.TimeoutError:
                        print("ASR: Timeout waiting for more audio")
                        audio_stream.closed = True
                        break
    except asyncio.TimeoutError:
        print(f"ASR: User inactive for {timeout} seconds")
    except Exception as e:
        print(f"ASR: Error: {e}")
        traceback.print_exc()
    finally:
        await output_queue.put(None)

async def baidu_llm(
    input_queue: asyncio.Queue, 
    output_queue: asyncio.Queue, 
    llm_client: object, 
    stop_event: asyncio.Event,
    interrupt_event: asyncio.Event,
    system=SYSTEM_PROMPT, 
    max_tokens=MAX_TOKENS,
    timeout=TIMEOUT,
    ) -> None:
    """
    LLM that generates texts based on the incoming trancribed message of the user
    Args:
        input_queue (asyncio.Queue): The queue to receive user input from ASR.
        output_queue (asyncio.Queue): The queue to send the generated response to TTS.
        llm_client (object): The LLM client to use for generating responses.
        stop_event (asyncio.Event): Event to signal when to stop the program.
        interrupt_event (asyncio.Event): Event to signal when to interrupt the current LLM generation.
        system (str, optional): The system prompt to use for the LLM, defaults to None.
        max_tokens (int, optional): The maximum number of tokens to generate, defaults to 512.
        timeout (float, optional): The timeout for the LLM request, defaults to 600 seconds.
    Returns:
        None
    Raises:
        Exception: If an error occurs during text generation.
    """
    try:
        texts = []
        assistants = []
        while not stop_event.is_set():
            text = await asyncio.wait_for(input_queue.get(), timeout=timeout)
            texts.append(text)
            if stop_event.is_set() or text is None:
                await output_queue.put(None)
                return 
            print(f"LLM: Generating response for: '{text}'")
            chat_completion = llm_client.generate(
                users=texts,
                system=system,
                assistants=assistants,
                max_tokens=max_tokens,
            )
            total_response = ""
            # Buffer for combining small chunks
            buffer = ""
            for chunk in chat_completion:
                response = chunk.choices[0].delta.content
                total_response += response
                prev = 0
                # split each response based on common puntuations.
                # Strip the uncommon symbols for each chunk.
                temp_text = ""
                for i in range(len(response)):
                    char = response[i]
                    if stop_event.is_set():
                        await output_queue.put(None)
                        return
                    if interrupt_event.is_set():
                        await output_queue.put(None)
                        break
                    # skip any other characters
                    if ucd.category(char).startswith('P') or ucd.category(char).startswith('S'):
                        if char in ['。', '，', '！', '？', '.', ',', '!', '?']:
                            # If punctuation, send the buffer
                            if i < prev + 10:
                                response = response[:i] + ' ' + response[i+1:]
                                continue
                            if buffer:
                                temp_text = buffer + response[prev:i]
                                buffer = ""
                            else:
                                temp_text = response[prev:i]
                            print(f"LLM: Sending text into TTS: '{temp_text}'")
                            await output_queue.put(temp_text)
                            prev = i + 1
                        else:
                            response = response[:i] + ' ' + response[i+1:]
                buffer += response[prev:]
                if interrupt_event.is_set():
                    print("LLM: Interrupt event set, stopping generation")
                    await output_queue.put(None)
                    break

            if interrupt_event.is_set():
                print("LLM: Interrupt event set, stopping generation")
                await output_queue.put(None)
            elif buffer:
                await output_queue.put(buffer)
            assistants.append(total_response)
            print(f"LLM: Generated {len(total_response)} characters")

    except Exception as e:
        print(f"LLM: Error: {e}")
        traceback.print_exc()
    finally:
        # Always send end marker
        await output_queue.put(None)
        print("LLM: Sent end marker")

async def local_tts(
    input_queue: asyncio.Queue, 
    output_queue: asyncio.Queue, 
    tts_model: object,
    stop_event: asyncio.Event = None,
    interrupt_event: asyncio.Event = None,
    ) -> None:
    """
    TTS model that runs locally, transcribing text in to speech with predefined model.
    Args:
        input_queue (asyncio.Queue): The queue to get input text from
        output_queue (asyncio.Queue): The queue to send generated audio to
        tts_model (object): The TTS model to use for generating audio
    Returns:
        None
    Raises:
        asyncio.TimeoutError: If a timeout occurs while waiting for input.
        Exception: If an error occurs during transcription.
    """
    print("TTS: Starting optimized transcription")
    total_samples_generated = 0
    
    try:
        text_buffer = ""
        
        while not stop_event.is_set():
            try:
                # Get text from LLM
                text = await asyncio.wait_for(input_queue.get(), timeout=TIMEOUT)
                print(f"TTS received text: {text}")
                if interrupt_event.is_set():
                    print("TTS: Interrupt event set, stopping transcription")
                    await output_queue.put(None)
                    break
                
                if stop_event.is_set():
                    print("TTS: Stop event set, stopping transcription")
                    await output_queue.put(None)
                    continue

                if text is None:
                    print("TTS: End of input")
                    # Process any remaining buffered text
                    if text_buffer.strip():
                        await process_text_chunk(text_buffer, output_queue, tts_model)
                    continue
                
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
                    if stop_event.is_set():
                        print("TTS: Stop event set, skipping sentence")
                        break
                    if interrupt_event.is_set():
                        print("TTS: Interrupt event set, skipping sentence")
                        break
                    samples = await process_text_chunk(sentence, output_queue, tts_model)
                    total_samples_generated += samples
                    
            except asyncio.TimeoutError:
                print("TTS: Timeout waiting for input")
                break
            except Exception as e:
                print(f"TTS: Error: {e}")
                traceback.print_exc()
                break
        
        print(f"TTS: Generated {total_samples_generated} total samples")
        
    finally:
        # Wait a bit for audio to be consumed
        await asyncio.sleep(0.5)
        # Send end marker
        await output_queue.put(None)
        print("TTS: Sent end marker")

async def process_text_chunk(text: str, output_queue: asyncio.Queue, tts_model: object) -> int:
    """
    Process a single text chunk and return number of samples generated.
    Args:
        text (str): The text to process.
        output_queue (asyncio.Queue): The queue to send generated audio to.
        tts_model (object): The TTS model to use for generating audio.
    Returns:
        int: The number of audio samples generated.
    Raises:
        Exception: If an error occurs during chunk processing.
    """
    if not text.strip():
        return 0
    
    print(f"TTS: Processing: '{text[:20]}...'")
    
    try:
        # Generate TTS audio
        audio_chunks = await tts_model.generate(text)
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
        chunk_size = AUDIO_CHUNK_SIZE  
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

async def azure_tts(
    input_queue: asyncio.Queue, 
    output_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    interrupt_event: asyncio.Event,
    timeout=TIMEOUT
) -> None:
    """
    Azure TTS
    Args:
        input_queue (asyncio.Queue): Queue for incoming text chunks from LLM.
        output_queue (asyncio.Queue): Queue for outgoing audio chunks to WebRTC peer.
        stop_event (asyncio.Event): Event to signal stopping the program.
        interrupt_event (asyncio.Event): Event to signal interrupting the TTS.
        timeout (float, optional): Timeout for waiting on input. Default to 600 seconds.
    Returns:
        None
    Raises:
        Exception: If an error occurs during processing.
    """
    total_samples_generated = 0
    sentence_count = 0
    
    try:
        current_tts_task = None
        while not stop_event.is_set():
            try:
                # Get text from LLM
                text = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                
                if stop_event.is_set():
                    print("TTS: Stopping")
                    break
                
                if interrupt_event.is_set():
                    print("TTS: Interrupted, clearing buffer")
                    # Clear output queue
                    try:
                        while True:
                            output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    continue
                
                if text is None:
                    print("TTS: End of input from LLM")
                    # Wait for current TTS to finish
                    if current_tts_task and not current_tts_task.done():
                        await current_tts_task
                    continue
                
                print(f"TTS: Received text: '{text[:50]}...'")
                sentences = text.split()
                
                for sentence in sentences:
                    if stop_event.is_set() or interrupt_event.is_set():
                        print("TTS: Interrupted during sentence processing")
                        break
                    
                    if sentence.strip():
                        sentence_count += 1
                        
                        # Wait for previous TTS to finish before starting new one
                        if current_tts_task and not current_tts_task.done():
                            await current_tts_task
                        
                        # Start new streaming TTS
                        current_tts_task = asyncio.create_task(
                            process_text(sentence.strip(), output_queue, stop_event, interrupt_event, sentence_count)
                        )
                        
                        # For the first sentence, wait a bit to get audio flowing
                        if sentence_count == 1:
                            try:
                                await asyncio.wait_for(asyncio.shield(current_tts_task), timeout=0.5)
                            except asyncio.TimeoutError:
                                pass  # Continue processing
                        
                        samples = await current_tts_task
                        total_samples_generated += samples
                
            except asyncio.TimeoutError:
                print("TTS: Timeout waiting for input")
                break
            except Exception as e:
                print(f"TTS: Error processing text: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"TTS: Completed - {sentence_count} sentences, {total_samples_generated} samples")
        
    finally:
        await output_queue.put(None)
        print("TTS: Sent end marker")


async def process_text(
    sentence: str, 
    output_queue: asyncio.Queue, 
    stop_event: asyncio.Event,
    interrupt_event: asyncio.Event,
    sentence_id: int,
    ) -> int:
    """
    Process TTS with Azure true streaming - output audio chunks as they arrive.
    Uses the synthesizing event for immediate audio streaming.
    """
    
    if stop_event.is_set() or interrupt_event.is_set():
        print(f"TTS: Sentence {sentence_id} skipped due to interruption")
        return

    print(f"TTS: Processing sentence {sentence_id}: '{sentence[:50]}...'")

    if not sentence.strip():
        return 0
    
    start_time = time.time()
    total_samples = 0
    first_chunk_time = None
    
    # Thread-safe containers
    audio_chunk_queue = queue.Queue()
    error_container = [None]
    synthesis_complete = threading.Event()
    
    try:
        print(f"TTS: Starting Azure streaming for sentence {sentence_id}: '{sentence[:50]}...'")
        
        def on_synthesizing(evt):
            """Handle audio chunks as they arrive - TRUE STREAMING"""
            nonlocal first_chunk_time, total_samples
            
            audio_data = evt.result.audio_data
            
            if interrupt_event.is_set() or stop_event.is_set():
                print(f"TTS: Interrupt event set during synthesis of sentence {sentence_id}")
                error_container[0] = "Interrupted or stopped"
                audio_chunk_queue.put(('error', "Interrupted or stopped"))
                synthesis_complete.set()
                return
            
            if audio_data and len(audio_data) > 0:
                # Track first chunk timing
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    print(f"TTS: First audio chunk for sentence {sentence_id} in {latency:.3f}s")
                
                # Put audio chunk in queue immediately
                audio_chunk_queue.put(('audio', audio_data))
                total_samples += len(audio_data) // 2  # 16-bit samples
                print(f"TTS: Streamed {len(audio_data)} bytes for sentence {sentence_id}")
        
        def on_synthesis_completed(evt):
            """Handle synthesis completion"""
            if evt.result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Log latency metrics if available
                if evt.result.properties:
                    first_byte_latency = evt.result.properties.get_property(
                        speechsdk.PropertyId.SpeechServiceResponse_SynthesisFirstByteLatencyMs
                    )
                    if first_byte_latency:
                        print(f"TTS: Sentence {sentence_id} - First byte latency: {first_byte_latency}ms")
            else:
                error_msg = f"Synthesis failed: {evt.result.reason}"
                error_container[0] = error_msg
                audio_chunk_queue.put(('error', error_msg))
            
            audio_chunk_queue.put(('done', None))
            print("===========================END EVENT SET HERE!!!==================================")
            synthesis_complete.set()
        
        def on_synthesis_cancelled(evt):
            """Handle synthesis cancellation"""
            error_msg = f"Synthesis cancelled: {evt.result.cancellation_details.error_details}"
            error_container[0] = error_msg
            audio_chunk_queue.put(('error', error_msg))
            synthesis_complete.set()
        
        # Connect event handlers for streaming
        azure_synthesizer.synthesizing.connect(on_synthesizing)
        azure_synthesizer.synthesis_completed.connect(on_synthesis_completed)
        azure_synthesizer.synthesis_canceled.connect(on_synthesis_cancelled)
        
        # Start synthesis (non-blocking)
        def start_synthesis():
            azure_synthesizer.speak_text_async(sentence)
        
        # Run synthesis in background thread
        synthesis_thread = threading.Thread(target=start_synthesis, daemon=True)
        synthesis_thread.start()
        
        # Process audio chunks as they arrive
        while not synthesis_complete.is_set() or not audio_chunk_queue.empty():
            try:
                # Get next chunk with short timeout
                chunk_type, chunk_data = audio_chunk_queue.get(timeout=0.1)
                
                if chunk_type == 'error':
                    print(f"TTS: Error in sentence {sentence_id}: {chunk_data}")
                    break
                elif chunk_type == 'done':
                    print(f"TTS: Streaming complete for sentence {sentence_id}")
                    break
                elif chunk_type == 'audio':
                    # Convert bytes to numpy array (16-bit PCM)
                    audio_np = np.frombuffer(chunk_data, dtype=np.int16)
                    
                    if len(audio_np) > 0:
                        # Apply audio improvements
                        #audio_np = apply_audio_improvements(audio_np)
                        
                        # Send immediately
                        await send_audio(audio_np, output_queue)
                
            except queue.Empty:
                # Continue waiting for more chunks
                await asyncio.sleep(0.01)
        
        # Disconnect event handlers
        azure_synthesizer.synthesizing.disconnect_all()
        azure_synthesizer.synthesis_completed.disconnect_all()
        azure_synthesizer.synthesis_canceled.disconnect_all()
        
        processing_time = time.time() - start_time
        print(f"TTS: Sentence {sentence_id} completed in {processing_time:.2f}s, {total_samples} samples")

        return total_samples
        
    except Exception as e:
        print(f"TTS: Error in Azure streaming TTS for sentence {sentence_id}: {e}")
        import traceback
        traceback.print_exc()
        return 0


async def send_audio(audio_np: np.ndarray, 
                     output_queue: asyncio.Queue, 
                     samples_per_frame: int=SAMPLES_PER_FRAME
                     ) -> None:
    """Send audio chunks immediately without waiting for complete sentence."""
    if len(audio_np) == 0:
        return

    chunk_size = samples_per_frame  # 10ms chunks
    
    for i in range(0, len(audio_np), chunk_size):
        chunk = audio_np[i:min(i + chunk_size, len(audio_np))]

        if len(chunk) < samples_per_frame:
            chunk = np.pad(chunk, (0, samples_per_frame - len(chunk)), 'constant')

        await output_queue.put(chunk)
        await asyncio.sleep(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal WebRTC audio logger")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8081, help="Port for HTTP server (default: 8081)")
    parser.add_argument("--mode", default="online", help="'local' to run locally; 'online' to call API")
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