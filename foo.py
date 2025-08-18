from config.imports import *
from config.constants import *

logging.basicConfig(filename=os.path.join(ROOT, f"logs/{time.time()}.log"), level=logging.INFO)
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
google_speech_end_timeout = duration_pb2.Duration(seconds=SILENCE_TIME)
google_asr_client = speech.SpeechClient(
    client_options=client_options.ClientOptions(
        api_endpoint=f"us-speech.googleapis.com"
    )
)
google_asr_config = speech.RecognitionConfig(
    auto_decoding_config=speech.AutoDetectDecodingConfig(),
    language_codes=[ASR_LANGUAGE],
    model="latest_long",
)
google_asr_streaming_feature = speech.StreamingRecognitionFeatures(
    enable_voice_activity_events=True,
    voice_activity_timeout=speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
        speech_end_timeout=google_speech_end_timeout
    )
)
google_asr_streaming_config = speech.StreamingRecognitionConfig(
    config=google_asr_config, 
    streaming_features=google_asr_streaming_feature,
)

# Resampler to change audio properties if needed
resampler = AudioResampler(rate=ASR_SAMPLE_RATE, layout=LAYOUT, format=FORMAT)
llm_client = BaiduClient(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
tts_model = EdgeTTS()

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

vad, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)

start_time = time.time()
llm_first_chunk_time = 0
llm_start_time = 0
tts_first_chunk_time = 0
tts_start_time = 0

# p = pa.PyAudio()
# stream = p.open(format=pa.paInt16, channels=1, rate=AUDIO_SAMPLE_RATE, output=True)

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
    Receive and audio frame from the track and 
    send it to another connected peer through a WebRTC connection.
    """
    kind = "audio"
    
    def __init__(self, audio_queue: asyncio.Queue, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        """
        Initialize the AudioPlayer with the given audio queue and sample rate.
        Args:
            audio_queue (asyncio.Queue): The queue to receive audio frames from.
            sample_rate (int, optional): The sample rate of the audio. 
                                         Defaults to AUDIO_SAMPLE_RATE = 24000.
        Returns:
            None
        """
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = sample_rate // 100
        
        # Use deque for efficient buffer management
        self._audio_buffer = deque()
        self._min_buffer = self.samples_per_frame * 100   # 1000ms minimum
        self._target_buffer = self.samples_per_frame * 200  # 2000ms target  
        self._max_buffer = self.samples_per_frame * 500   # 5000ms maximum
        
        # State tracking
        self._started = False
        self._ending = False
        self._ended = False
        self._frames_sent = 0
        self._total_samples_received = 0
        self._total_samples_played = 0
        self._consecutive_silence = 0
        
        self._last_frame_time = time.time()
        # 0.01 seconds for 240 samples at 24kHz
        self._frame_interval = self.samples_per_frame / self.sample_rate  

        print(f"AudioPlayer: Initialized with large buffers - min:{self._min_buffer}, "
              f"target:{self._target_buffer}, max:{self._max_buffer}")

    async def recv(self) -> AudioFrame:
        """
        Generate audio frames and send them to the connected peer.
        Args:
            self: The class instance.
        Returns:
            frame (AudioFrame): The generated audio frame.
        """
        # Control frame timing first
        await self._control_frame_timing()
        
        # Fill buffer aggressively when empty
        await self._fill_buffer_aggressively()
        
        # Generate output frame
        if len(self._audio_buffer) >= self.samples_per_frame:
            # Extract exactly samples_per_frame from buffer
            frame_data = [self._audio_buffer.popleft() for _ in range(self.samples_per_frame)]
            audio_array = np.array(frame_data, dtype=np.int16).reshape(1, -1)
            self._consecutive_silence = 0
            self._total_samples_played += self.samples_per_frame
            
            # Less frequent logging
            if self._frames_sent % 50 == 0:
                print(f"AudioPlayer: Frame {self._frames_sent}, buffer={len(self._audio_buffer)}, "
                      f"played={self._total_samples_played}, received={self._total_samples_received}")

        else:
            # Buffer underrun - send silence
            audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
            self._consecutive_silence += 1
            
            # If we're ending and buffer is empty, mark as ended
            if self._ending and len(self._audio_buffer) == 0:
                if not self._ended:
                    self._ended = True
                    print(f"AudioPlayer: Playback complete. Frames: {self._frames_sent}, "
                          f"Played: {self._total_samples_played}, "
                          f"Received: {self._total_samples_received}")

            # Log underruns but not too frequently
            # if self._consecutive_silence == 1:
            #     print(f"AudioPlayer: Buffer underrun at frame {self._frames_sent}, "
            #           f"buffer: {len(self._audio_buffer)}, "
            #           f"received: {self._total_samples_received}")
            # elif self._consecutive_silence % 100 == 0:
            #     print(f"AudioPlayer: Extended underrun - {self._consecutive_silence} "
            #           f"consecutive silence frames")

        # Create frame
        frame = AudioFrame.from_ndarray(audio_array, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        
        self._timestamp += self.samples_per_frame
        self._frames_sent += 1
        
        return frame
    
    async def _control_frame_timing(self):
        """
        Control frame timing to match WebRTC expectations (10ms intervals).
        """
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        
        # WebRTC expects frames every 10ms (0.01 seconds)
        target_interval = self._frame_interval
        
        if elapsed < target_interval:
            sleep_time = target_interval - elapsed
            if sleep_time > 0.001:  # Only sleep if meaningful
                await asyncio.sleep(sleep_time)
        
        self._last_frame_time = time.time()
    
    async def _fill_buffer_aggressively(self):
        """
        Fill buffer aggressively to handle large incoming audio streams.
        """
        # Don't wait if we have enough audio or we're ending
        if len(self._audio_buffer) >= self._min_buffer and self._started:
            return
        
        # Fill more aggressively when buffer is low
        fill_attempts = 0
        max_attempts = 20 if len(self._audio_buffer) < self.samples_per_frame * 10 else 10
        
        while (len(self._audio_buffer) < self._target_buffer and 
               not self._ending and 
               fill_attempts < max_attempts):
            
            try:
                # Much longer timeout to wait for audio
                timeout = 0.1 if not self._started else 0.05
                
                audio_data = await asyncio.wait_for(
                    self.audio_queue.get(), 
                    timeout=timeout
                )
                
                if audio_data is None:
                    # End marker received
                    self._ending = True
                    print(f"AudioPlayer: End marker received, "
                          f"buffer has {len(self._audio_buffer)} samples")
                    print(f"AudioPlayer: Total received: {self._total_samples_received}, "
                          f"Total played: {self._total_samples_played}")
                    break
                
                # Add to buffer with proper tracking
                if isinstance(audio_data, np.ndarray):
                    data_list = audio_data.tolist()
                else:
                    data_list = list(audio_data)
                
                # Track received samples
                self._total_samples_received += len(data_list)
                
                # Prevent buffer overflow but don't drop audio unnecessarily
                available_space = self._max_buffer - len(self._audio_buffer)
                if available_space > 0:
                    data_to_add = data_list[:available_space]
                    self._audio_buffer.extend(data_to_add)
                    
                    if len(data_list) > available_space:
                        dropped = len(data_list) - available_space
                        print(f"AudioPlayer: Buffer overflow - dropped {dropped} "
                              f"samples (buffer: {len(self._audio_buffer)})")
                else:
                    print(f"AudioPlayer: Buffer full ({len(self._audio_buffer)}), "
                          f"dropping {len(data_list)} samples")

                fill_attempts += 1
                
                # Start playback when we have enough buffer
                if not self._started and len(self._audio_buffer) >= self._target_buffer:
                    self._started = True
                    print(f"AudioPlayer: Starting playback with "
                          f"{len(self._audio_buffer)} samples buffered")

                # Log progress for large buffers
                if fill_attempts % 10 == 0:
                    print(f"AudioPlayer: Filled {fill_attempts} chunks, "
                          f"buffer now: {len(self._audio_buffer)}")

            except asyncio.TimeoutError:
                # If we haven't started yet, keep waiting
                if not self._started and not self._ending:
                    continue
                break
    
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
    Optimized offer handler with better resource management.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = f"PC-{uuid.uuid4().hex[:8]}"
    pcs.add(pc)

    # print(f"\n{'='*50}")
    # print(f"{pc_id}: New connection from {request.remote}")
    # print(f"{'='*50}")

    # Create session-specific queues
    asr_queue = asyncio.Queue(maxsize=50)
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
                print(f"{pc_id}: Starting ASR")
                text = await google_asr(track, stop_event, output_queue=asr_queue)

                if not text or text.strip() == "":
                    print(f"{pc_id}: Empty ASR result")
                    await audio_queue.put(None)
                    return
                
                print(f"{pc_id}: ASR result: '{text}'")
                
                # Start pipeline tasks of LLM and TTS
                pc._llm_task = asyncio.create_task(
                    baidu_llm(asr_queue, llm_queue, llm_client, SYSTEM_PROMPT, MAX_TOKENS)
                )
                pc._tts_task = asyncio.create_task(
                    online_tts(llm_queue, audio_queue)
                )
                
                print(f"{pc_id}: Pipeline started")
                
            except Exception as e:
                print(f"{pc_id}: Pipeline error: {e}")
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

def energy(audio):
    return np.mean(np.square(audio))

async def google_asr(
    track: AudioStreamTrack, 
    stop_event: asyncio.Event, 
    output_queue: asyncio.Queue
) -> None:
    """
    Google ASR with VAD-based speech segmentation.
    Relies entirely on Google's VAD for speech detection - no energy-based filtering.
    
    Features:
    1. Graceful termination when stop button is pressed
    2. Automatic stream reopening for new speech after silence
    3. Uses Google's built-in VAD events for accurate speech detection
    
    Args:
        track: Audio track to transcribe
        stop_event: Event to signal when to stop transcription
        output_queue: Queue to put completed transcripts for LLM
    """
    
    try:
        # Main loop - continues until stop_event is set
        while not stop_event.is_set():
            # Create new session for each speech segment
            audio_queue = queue.Queue()
            session_output_queue = queue.Queue()
            session_active = threading.Event()
            session_active.set()
            
            # Track VAD state
            speech_started = False
            vad_timeout = False
            
            # Create audio stream for this session
            audio_stream = AudioStream(
                audio_queue, 
                rate=ASR_SAMPLE_RATE, 
                chunk_size=ASR_CHUNK_SIZE, 
                language_code=ASR_LANGUAGE,
            )
            
            def run_google_stream():
                """Run Google ASR for this session with VAD monitoring"""
                nonlocal speech_started, vad_timeout
                transcript = ""
                interim_transcript = ""
                
                try:
                    with audio_stream:
                        # Create request generator
                        def request_generator():
                            # First request with config
                            yield speech.StreamingRecognizeRequest(
                                recognizer=f"projects/{GOOGLE_PROJ_ID}/locations/us/recognizers/crick-stt",
                                streaming_config=google_asr_streaming_config,
                            )
                            
                            # Then audio requests
                            for content in audio_stream.generate():
                                if content:
                                    yield speech.StreamingRecognizeRequest(
                                        recognizer=f"projects/{GOOGLE_PROJ_ID}/locations/us/recognizers/crick-stt",
                                        audio=content
                                    )
                        
                        # Process responses
                        responses = google_asr_client.streaming_recognize(
                            requests=request_generator()
                        )
                        
                        for response in responses:
                            # Check for voice activity events
                            if hasattr(response, 'speech_event_type'):
                                event_type = response.speech_event_type
                                
                                if event_type == speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN:
                                    speech_started = True
                                    print("VAD: Speech activity BEGIN - person started talking")
                                    
                                elif event_type == speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END:
                                    print("VAD: Speech activity END - person stopped talking")
                                    # Person stopped talking, but we might wait for more speech
                                    # or for the timeout to finalize
                                    
                                elif event_type == speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_TIMEOUT:
                                    print(f"VAD: Speech TIMEOUT - {SILENCE_TIME}s of silence reached")
                                    vad_timeout = True
                                    # This is our signal to close the session and process the transcript
                                    session_active.clear()
                                    break
                            
                            # Process transcription results
                            if response.results:
                                for result in response.results:
                                    if result.alternatives:
                                        current_transcript = result.alternatives[0].transcript
                                        
                                        if result.is_final:
                                            transcript = current_transcript
                                            print(f"ASR: Final transcript: '{transcript}'")
                                        else:
                                            interim_transcript = current_transcript
                                            # Optionally print interim results for debugging
                                            # print(f"ASR: Interim: '{interim_transcript}'")
                    
                    # Use final transcript if available, otherwise use last interim
                    final_result = transcript if transcript else interim_transcript
                    session_output_queue.put(final_result)
                    
                except Exception as e:
                    print(f"ASR thread error: {e}")
                    session_output_queue.put("")
                finally:
                    session_active.clear()
            
            # Start ASR thread for this session
            asr_thread = threading.Thread(target=run_google_stream, daemon=True)
            
            frame = await track.recv()
            print('=' * 50)
            print("Google ASR: Starting with VAD")
            print('=' * 50)
            asr_thread.start()
            
            # Feed audio to ASR until VAD timeout or stop is pressed
            # We simply pass ALL audio to Google and let it handle the VAD
            while session_active.is_set() and not stop_event.is_set():
                try:
                    # Receive audio frame
                    frame = await asyncio.wait_for(track.recv(), timeout=0.5)
                    
                    # Resample and prepare audio
                    frame = resampler.resample(frame)[0].to_ndarray()
                    frame = np.squeeze(frame, axis=0).astype(np.int16)
                    
                    # Send ALL audio to Google - let Google's VAD handle detection
                    audio_bytes = frame.tobytes()
                    audio_queue.put(audio_bytes)
                    
                except asyncio.TimeoutError:
                    # Timeout is fine - Google will handle silence detection
                    # Just check if we should continue
                    if not session_active.is_set():
                        break
                    continue
                    
                except MediaStreamError:
                    print("ASR: Media stream ended")
                    stop_event.set()
                    break
            
            # Signal ASR thread to stop
            audio_queue.put(None)
            
            # Wait for ASR thread to complete with timeout
            asr_thread.join(timeout=5.0)
            
            # Get transcription result
            try:
                transcript = session_output_queue.get_nowait() if not session_output_queue.empty() else ""
                if transcript.strip():
                    await output_queue.put(transcript.strip())
                    print(f"ASR: Queued transcript for LLM: '{transcript.strip()}'")
                    print('=' * 30)
            except queue.Empty:
                print("ASR: No transcript from session")
            
            # If stop event is set, break the main loop
            if stop_event.is_set():
                print("ASR: Stop event detected - terminating")
                break
            
            # Check why session ended
            if vad_timeout:
                print("ASR: Session ended due to VAD timeout - ready for next speech")
                # Reset for next session
                vad_timeout = False
            elif not speech_started:
                print("ASR: No speech detected in session - continuing to listen")
                # Small delay to prevent tight loop when no speech
                await asyncio.sleep(0.1)
            else:
                print("ASR: Session ended - ready for next segment")
            
            # Small delay before starting next session
            await asyncio.sleep(0.05)
    
    except Exception as e:
        logger.error(f"ASR: Fatal error: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    finally:
        # Ensure we send end signal to LLM
        await output_queue.put(None)
        print("ASR: Cleanup complete - sent end signal to output queue")

# ALTERNATIVE: If you want to save partial generations in history
async def baidu_llm(
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    llm_client: BaiduClient,
    system_prompt=None,
    max_tokens=512
    ) -> None:
    """
    ALTERNATIVE: Save even partial generations to conversation history.
    """
    users = []
    assistants = []
    current_generation_task = None
    partial_generation = {"text": "", "active": False}
    
    while True:
        try:
            text = await input_queue.get()
            
            if text is None:
                # Handle shutdown
                if current_generation_task and not current_generation_task.done():
                    current_generation_task.cancel()
                    try:
                        await current_generation_task
                    except asyncio.CancelledError:
                        pass
                break
            
            print(f"LLM: New input: '{text[:50]}...'")
            
            # CHANGED: Save partial generation before cancelling
            if current_generation_task and not current_generation_task.done():
                print("LLM: Saving partial generation and starting new one...")
                current_generation_task.cancel()
                
                try:
                    await current_generation_task
                except asyncio.CancelledError:
                    # Save whatever was generated so far
                    if partial_generation["text"].strip():
                        assistants.append(partial_generation["text"])
                        print(f"LLM: Saved partial: '{partial_generation['text'][:50]}...'")
                
                # Reset partial generation tracker
                partial_generation = {"text": "", "active": False}
            
            users.append(text)
            
            # Start new generation
            current_generation_task = asyncio.create_task(
                llm_helper(
                    users, assistants, output_queue, llm_client, 
                    system_prompt, max_tokens, partial_generation
                )
            )
            
            try:
                await current_generation_task
                # If completed successfully, partial_generation will be handled inside
            except asyncio.CancelledError:
                print("LLM: Generation cancelled for new input")
                
        except Exception as e:
            print(f"LLM: Error: {e}")

async def llm_helper(
    users: list,
    assistants: list,
    output_queue: asyncio.Queue,
    llm_client: BaiduClient,
    system_prompt: str,
    max_tokens: int,
    partial_generation: dict
) -> None:
    """
    Generation with partial text tracking for cancellation.
    """
    partial_generation["active"] = True
    partial_generation["text"] = ""
    
    try:
        chat_completion = llm_client.generate(
            users=users,
            assistants=assistants,
            system=system_prompt,
            max_tokens=max_tokens,
        )
        
        buffer = ""
        
        for chunk in chat_completion:
            if asyncio.current_task().cancelled():
                await output_queue.put(None)
                raise asyncio.CancelledError()
            
            response = chunk.choices[0].delta.content
            partial_generation["text"] += response  # Track all generated text
            
            # Process and send chunks as before
            prev = 0
            for i, char in enumerate(response):
                if char != '\n':
                    if char in ['。', '，', '！', '？', '.', ',', '!', '?'] and i > prev + 10:
                        if asyncio.current_task().cancelled():
                            await output_queue.put(None)
                            raise asyncio.CancelledError()
                        
                        if buffer:
                            temp_text = buffer + response[prev:i+1]
                            buffer = ""
                        else:
                            temp_text = response[prev:i+1]
                        
                        await output_queue.put(temp_text)
                        prev = i + 1
                else:
                    prev = i + 1
            
            buffer += response[prev:]
        
        if buffer:
            await output_queue.put(buffer)
        
        # Completed successfully
        assistants.append(partial_generation["text"])
        print(f"LLM: Completed and saved: '{partial_generation['text'][:50]}...'")
        
        await output_queue.put(None)
        partial_generation["active"] = False
        
    except asyncio.CancelledError:
        print(f"LLM: Cancelled - partial text will be saved by caller")
        partial_generation["active"] = False
        raise

async def local_tts(
    input_queue: asyncio.Queue, 
    output_queue: asyncio.Queue, 
    tts_model: object
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
        chunk_size = AUDIO_CHUNK_SIZE  # 2400 samples = 100ms
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

async def online_tts(input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    """
    Azure TTS with true streaming - output audio chunks as they arrive.
    
    Key optimizations:
    1. True streaming using Azure's synthesizing event
    2. Minimal buffering for lowest latency
    3. Pre-connected synthesizer for faster response
    4. Aggressive sentence splitting for better responsiveness
    """
    total_samples_generated = 0
    sentence_count = 0
    
    try:
        text_buffer = ""
        current_tts_task = None
        
        while True:
            try:
                # Get text from LLM
                text_chunk = await asyncio.wait_for(input_queue.get(), timeout=30.0)
                
                if text_chunk is None:
                    print("TTS: End of input from LLM")
                    # Wait for current TTS to finish
                    if current_tts_task and not current_tts_task.done():
                        await current_tts_task
                    
                    # Process any remaining text
                    if text_buffer.strip():
                        sentence_count += 1
                        samples = await process_azure_streaming_tts(
                            text_buffer.strip(), output_queue, sentence_count
                        )
                        total_samples_generated += samples
                    break
                
                print(f"TTS: Received text chunk: '{text_chunk[:50]}...'")
                text_buffer += text_chunk
                
                # Use aggressive sentence splitting for lower latency
                sentences, text_buffer = extract_sentences_aggressive(text_buffer)
                
                for sentence in sentences:
                    if sentence.strip():
                        sentence_count += 1
                        
                        # Wait for previous TTS to finish before starting new one
                        if current_tts_task and not current_tts_task.done():
                            await current_tts_task
                        
                        # Start new streaming TTS
                        current_tts_task = asyncio.create_task(
                            process_azure_streaming_tts(sentence.strip(), output_queue, sentence_count)
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


async def process_azure_streaming_tts(sentence: str, output_queue: asyncio.Queue, sentence_id: int) -> int:
    """
    Process TTS with Azure true streaming - output audio chunks as they arrive.
    Uses the synthesizing event for immediate audio streaming.
    """
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
                        audio_np = apply_audio_improvements(audio_np)
                        
                        # Send immediately
                        await send_audio_immediately(audio_np, output_queue)
                
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


async def send_audio_immediately(audio_np: np.ndarray, output_queue: asyncio.Queue):
    """Send audio chunks immediately without waiting for complete sentence."""
    if len(audio_np) == 0:
        return
    
    # Use smaller chunks for lower latency (assuming SAMPLES_PER_FRAME = 240 for 24kHz at 10ms)
    SAMPLES_PER_FRAME = 240  # Adjust based on your constants
    chunk_size = SAMPLES_PER_FRAME * 5  # 50ms chunks
    
    for i in range(0, len(audio_np), chunk_size):
        chunk = audio_np[i:min(i + chunk_size, len(audio_np))]
        
        # Pad if necessary
        if len(chunk) < SAMPLES_PER_FRAME:
            chunk = np.pad(chunk, (0, SAMPLES_PER_FRAME - len(chunk)), 'constant')
        
        await output_queue.put(chunk)
        
        # Minimal delay to prevent overwhelming
        await asyncio.sleep(0.001)


def extract_sentences_aggressive(text_buffer: str) -> tuple:
    """
    More aggressive sentence extraction for lower latency.
    
    Splits on commas and other punctuation, not just sentence endings.
    This allows audio to start playing sooner.
    """
    if not text_buffer.strip():
        return [], ""
    
    sentences = []
    
    # Split on various punctuation for more responsive streaming
    parts = re.split(r'([.!?。！？，,；;：:]+)', text_buffer)
    
    current_sentence = ""
    i = 0
    
    while i < len(parts) - 1:
        current_sentence += parts[i]
        
        if i + 1 < len(parts) and re.match(r'[.!?。！？，,；;：:]+', parts[i + 1]):
            current_sentence += parts[i + 1]
            
            # Only create sentence if it's long enough
            if len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
            
            i += 2
        else:
            i += 1
    
    # Handle remaining text
    remaining_parts = parts[i:]
    remaining = current_sentence + ''.join(remaining_parts)
    
    print(f"TTS: Aggressive split - {len(sentences)} chunks, remaining: '{remaining[:30]}...'")
    return sentences, remaining.strip()


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
