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

# Queues to streamingly store LLM output and TTS audio output
llm_tts_queue = asyncio.Queue()
tts_output_queue = asyncio.Queue()

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
tts_model = EdgeTTS()

# Google Cloud clients for TTS (online)
google_tts_client = texttospeech.TextToSpeechClient()
# See https://cloud.google.com/text-to-speech/docs/voices for all voices.
google_tts_streaming_config = texttospeech.StreamingSynthesizeConfig(
    voice=texttospeech.VoiceSelectionParams(
        name=TTS_VOICE,
        language_code=TTS_LANGUAGE,
    )
)
# Set the config for your stream. The first request must contain your config, 
# and then each subsequent request must contain text.
google_tts_config_request = texttospeech.StreamingSynthesizeRequest(
    streaming_config=google_tts_streaming_config
)

start_time = time.time()
llm_first_chunk_time = 0
llm_start_time = 0
tts_first_chunk_time = 0
tts_start_time = 0


# p = pa.PyAudio()
# stream = p.open(format=pa.paInt16, channels=1, rate=AUDIO_SAMPLE_RATE, output=True)

class VAD:
    """
     Voice Activity Detection with better speech detection.
    """
    def __init__(self, threshold=VAD_THRESHOLD, speech_frames_required=3):
        self.threshold = threshold
        self.speech_frames_required = speech_frames_required
        self.speech_frame_count = 0
        self.is_currently_speaking = False
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
    A simple Voice Activity Detection (VAD) class.
    """
    def __init__(self, threshold=VAD_THRESHOLD):
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Detect if frame contains speech.
        """
        if len(frame) == 0:
            return False

        rms_energy = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        return rms_energy > self.threshold

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
    """
     AudioPlayer with proper interruption support and buffer management.
    """
    kind = "audio"
    
    def __init__(self, audio_queue: asyncio.Queue, sample_rate: int = 24000) -> None:
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = sample_rate // 100  # 10ms frames
        
        # Simplified buffer management
        self._audio_buffer = deque()
        self._max_buffer_ms = 500  # Maximum 500ms of audio buffered
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
        """Generate audio frames with proper interruption support."""
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
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        
        self._timestamp += self.samples_per_frame
        self._frames_sent += 1
        
        return frame
    
    async def _control_timing(self):
        """Control frame timing for consistent 10ms intervals."""
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
            # No audio available, continue
            pass
    
    def request_interrupt(self):
        """Request immediate interruption of audio playback."""
        self._interrupt_requested = True
        print("AudioPlayer: Interrupt requested")
    
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
    Offer handler with proper interruption support.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PC-{uuid.uuid4().hex[:8]}"
    pcs.add(pc)

    print(f"{pc_id}: New connection established")

    # Create session-specific queues
    asr_queue = asyncio.Queue(maxsize=20)
    llm_queue = asyncio.Queue(maxsize=20)
    audio_queue = asyncio.Queue(maxsize=100)  # Smaller queue for better responsiveness
    
    # Create  components
    recorder = MediaBlackhole()
    audio_player = AudioPlayer(audio_queue, sample_rate=24000)
    _vad = SimpleVAD()
    
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
            
            # Create control events
            stop_event = asyncio.Event()
            interrupt_event = asyncio.Event()
            pc._stop_event = stop_event
            pc._interrupt_event = interrupt_event
            
            try:
                # Start  pipeline
                pc._asr_task = asyncio.create_task(
                    google_asr(
                        track, stop_event, interrupt_event, 
                        asr_queue, audio_player, vad=_vad
                    )
                )

                pc._llm_task = asyncio.create_task(
                    llm_generator(
                        asr_queue, llm_queue, llm_client, 
                        SYSTEM_PROMPT, MAX_TOKENS, 
                        stop_event, interrupt_event
                    )
                )
                
                pc._tts_task = asyncio.create_task(
                    tts_pipeline(
                        llm_queue, audio_queue, 
                        stop_event, interrupt_event
                    )
                )
                
                print(f"{pc_id}:  pipeline started")
                
            except Exception as e:
                print(f"{pc_id}: Pipeline error: {e}")
                import traceback
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

async def local_asr(track: AudioStreamTrack, stop_event: asyncio.Event = None) -> str:
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

async def google_asr(
    track: AudioStreamTrack, 
    stop_event: asyncio.Event, 
    interrupt_event: asyncio.Event,
    output_queue: asyncio.Queue,
    audio_player: AudioPlayer,
    vad=None,
    vad_threshold=VAD_THRESHOLD,
    sample_rate=ASR_SAMPLE_RATE,
    chunk_size=ASR_CHUNK_SIZE,
    language_code=ASR_LANGUAGE,
    timeout=TIMEOUT,
) -> None:
    """
     ASR with proper interruption handling.
    """
    try:
        resampler = AudioResampler(rate=sample_rate, layout='mono', format='s16')
        
        while not stop_event.is_set():
            frame = await asyncio.wait_for(track.recv(), timeout=timeout)
            frame = resampler.resample(frame)[0].to_ndarray()
            frame = np.squeeze(frame, axis=0).astype(np.int16)
            
            # Check for speech with  VAD
            if not vad or vad.is_speech(frame):
                # IMMEDIATELY interrupt audio playback when speech detected
                audio_player.request_interrupt()
                interrupt_event.set()
                
                print("ASR: Speech detected - starting transcription")
                
                # Set up streaming recognition
                audio_queue = queue.Queue()
                session_output_queue = queue.Queue(maxsize=1)
                
                audio_stream = AudioStream(
                    audio_queue, 
                    rate=sample_rate, 
                    chunk_size=chunk_size,
                    language_code=language_code
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
        import traceback
        traceback.print_exc()
    finally:
        await output_queue.put(None)

async def llm_generator(
    input_queue: asyncio.Queue, 
    output_queue: asyncio.Queue, 
    llm_client: object, 
    system_prompt=None, 
    max_tokens=512,
    stop_event: asyncio.Event=None,
    interrupt_event: asyncio.Event=None,
    timeout=TIMEOUT,
    ) -> None:
    """
    LLM that generates texts based on the incoming trancribed message of the user
    Args:
        text (str): The input text to generate a response for, also known as the user input
        output_queue (asyncio.Queue): The queue to send the generated response to
        llm_client (object): The LLM client to use for generating responses
        system_prompt (str, optional): The system prompt to use for the LLM, defaults to None
        max_tokens (int, optional): The maximum number of tokens to generate, defaults to 512
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
            llm_start_time = time.time()
                # Stream the response
            chat_completion = llm_client.generate(
                users=texts,
                system=system_prompt,
                assistants=assistants,
                max_tokens=max_tokens,
            )
            total_response = ""
            # Buffer for combining small chunks
            buffer = ""
            for chunk in chat_completion:
                response = chunk.choices[0].delta.content
                total_response += response
                # if init:
                #     init_text = response[:10] if len(response) > 10 else response
                #     await output_queue.put(init_text)
                #     response = response[10:] if len(response) > 10 else ""
                #     init = False
                    
                #print(response, end='', flush=True)
                prev = 0
                # split each response based on common puntuations.
                # Strip the uncommon symbols for each chunk.
                for i, char in enumerate(response):
                    if stop_event.is_set():
                        await output_queue.put(None)
                        return
                    if interrupt_event.is_set():
                        await output_queue.put(None)
                        break
                    # skip newlines
                    if char != '\n':
                        if char in ['。', '，', '！', '？', '.', ',', '!', '?'] and i > prev + 10:
                            # If punctuation, send the buffer
                            if buffer:
                                temp_text = buffer + response[prev:i+1]
                                buffer = ""
                            else:
                                temp_text = response[prev:i+1]
                            #temp_text = temp_text.strip(' \n*\"‘’“”；：')
                            await output_queue.put(temp_text)
                            print(f"\ntemp_text in LLM is {temp_text}")
                            prev = i + 1
                    else:
                        prev = i + 1
                buffer += response[prev:]
                #buffer.strip(' \n*\"‘’“”；：')
                global llm_first_chunk_time
                llm_first_chunk_time = time.time()
                print(f"LLM: First chunk took {llm_first_chunk_time - llm_start_time:.2f} seconds")

                print(f"\nbuffer in LLM is {buffer}")
                #await asyncio.sleep(5)
                
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
        import traceback
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

async def tts_pipeline(
    input_queue: asyncio.Queue, 
    output_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    interrupt_event: asyncio.Event,
    timeout=30
) -> None:
    """
     TTS pipeline with proper interruption handling.
    """
    try:
        text_buffer = ""
        current_sentence_id = 0
        
        while not stop_event.is_set():
            try:
                # Get text from LLM
                text_chunk = await asyncio.wait_for(input_queue.get(), timeout=timeout)
                
                if stop_event.is_set():
                    print("TTS: Stopping")
                    break
                
                if interrupt_event.is_set():
                    print("TTS: Interrupted, clearing buffer")
                    text_buffer = ""
                    # Clear output queue
                    try:
                        while True:
                            output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    continue
                
                if text_chunk is None:
                    print("TTS: End of input")
                    # Process remaining buffer
                    if text_buffer.strip():
                        current_sentence_id += 1
                        await process_text_with_interruption(
                            text_buffer.strip(), 
                            output_queue, 
                            stop_event,
                            interrupt_event, 
                            current_sentence_id
                        )
                    break
                
                text_buffer += text_chunk
                
                # Extract complete sentences
                sentences, text_buffer = extract_complete_sentences(text_buffer)
                
                for sentence in sentences:
                    if stop_event.is_set() or interrupt_event.is_set():
                        print("TTS: Interrupted during sentence processing")
                        break
                    
                    if sentence.strip():
                        current_sentence_id += 1
                        await process_text_with_interruption(
                            sentence.strip(), 
                            output_queue, 
                            stop_event,
                            interrupt_event, 
                            current_sentence_id
                        )
                
            except asyncio.TimeoutError:
                print("TTS: Timeout waiting for input")
                break
            except Exception as e:
                print(f"TTS: Error: {e}")
                break
    
    finally:
        await output_queue.put(None)
        print("TTS: Pipeline ended")


async def process_text_with_interruption(
    text: str, 
    output_queue: asyncio.Queue, 
    stop_event: asyncio.Event,
    interrupt_event: asyncio.Event, 
    sentence_id: int
) -> None:
    """
    Process TTS text with interruption checking.
    """
    if stop_event.is_set() or interrupt_event.is_set():
        print(f"TTS: Sentence {sentence_id} skipped due to interruption")
        return
    
    print(f"TTS: Processing sentence {sentence_id}: '{text[:50]}...'")
    
    try:
        # Create streaming request
        def request_generator():
            yield google_tts_config_request
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )
        
        # Process TTS with interruption checking
        import queue
        audio_chunk_queue = queue.Queue()
        
        def run_streaming_tts():
            try:
                streaming_responses = google_tts_client.streaming_synthesize(request_generator())
                for response in streaming_responses:
                    if interrupt_event.is_set():
                        print(f"TTS: Sentence {sentence_id} interrupted during generation")
                        break
                    if response.audio_content:
                        audio_chunk_queue.put(('audio', response.audio_content))
                audio_chunk_queue.put(('done', None))
            except Exception as e:
                audio_chunk_queue.put(('error', str(e)))
        
        import threading
        tts_thread = threading.Thread(target=run_streaming_tts, daemon=True)
        tts_thread.start()
        
        # Process audio chunks with interruption checking
        while not interrupt_event.is_set():
            try:
                chunk_type, chunk_data = audio_chunk_queue.get(timeout=5.0)
                
                if chunk_type == 'done':
                    break
                elif chunk_type == 'error':
                    print(f"TTS: Error in sentence {sentence_id}: {chunk_data}")
                    break
                elif chunk_type == 'audio':
                    if interrupt_event.is_set():
                        print(f"TTS: Sentence {sentence_id} interrupted during audio processing")
                        break
                    
                    audio_np = np.frombuffer(chunk_data, dtype=np.int16)
                    if len(audio_np) > 0:
                        await send_audio_chunks(audio_np, output_queue, interrupt_event)
                
            except queue.Empty:
                print(f"TTS: Timeout waiting for audio in sentence {sentence_id}")
                break
        
        if interrupt_event.is_set():
            print(f"TTS: Sentence {sentence_id} was interrupted")
        else:
            print(f"TTS: Sentence {sentence_id} completed")
            
    except Exception as e:
        print(f"TTS: Error processing sentence {sentence_id}: {e}")


async def send_audio_chunks(audio_np: np.ndarray, output_queue: asyncio.Queue, interrupt_event: asyncio.Event):
    """Send audio chunks with interruption checking."""
    chunk_size = 240  # 10ms chunks at 24kHz
    
    for i in range(0, len(audio_np), chunk_size):
        if interrupt_event.is_set():
            break
        
        chunk = audio_np[i:min(i + chunk_size, len(audio_np))]
        if len(chunk) < chunk_size:
            # Pad last chunk
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        await output_queue.put(chunk)
        await asyncio.sleep(0.001)  # Small delay to prevent overwhelming


def extract_complete_sentences(text: str) -> tuple:
    """Extract complete sentences for processing."""
    import re
    
    # Split on sentence endings
    sentences = re.split(r'([.!?。！？]+)', text)
    
    complete_sentences = []
    remaining = ""
    
    i = 0
    while i < len(sentences) - 1:
        sentence = sentences[i]
        if i + 1 < len(sentences) and re.match(r'[.!?。！？]+', sentences[i + 1]):
            sentence += sentences[i + 1]
            if sentence.strip():
                complete_sentences.append(sentence.strip())
            i += 2
        else:
            remaining += sentence
            i += 1
    
    # Add any remaining parts
    if i < len(sentences):
        remaining += sentences[i]
    
    return complete_sentences, remaining

def apply_audio_improvements(audio_np: np.ndarray) -> np.ndarray:
    """Apply basic audio improvements."""
    if len(audio_np) == 0:
        return audio_np
    
    # Remove DC offset
    if len(audio_np) > 1:
        audio_np = audio_np - np.mean(audio_np)
    
    # Simple clipping prevention
    max_val = np.max(np.abs(audio_np))
    if max_val > 30000:
        compression_ratio = 30000 / max_val
        audio_np = (audio_np * compression_ratio).astype(np.int16)
    
    return audio_np

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