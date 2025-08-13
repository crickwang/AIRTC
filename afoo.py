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
resampler = AudioResampler(rate=AUDIO_SAMPLE_RATE, layout=LAYOUT, format=FORMAT)
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
            
            # # Less frequent logging
            # if self._frames_sent % 50 == 0:
            #     print(f"AudioPlayer: Frame {self._frames_sent}, buffer={len(self._audio_buffer)}, "
            #           f"played={self._total_samples_played}, received={self._total_samples_received}")

        else:
            # Buffer underrun - send silence
            audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
            self._consecutive_silence += 1
            
            # If we're ending and buffer is empty, mark as ended
            if self._ending and len(self._audio_buffer) == 0:
                if not self._ended:
                    self._ended = True
                    # print(f"AudioPlayer: Playback complete. Frames: {self._frames_sent}, "
                    #       f"Played: {self._total_samples_played}, "
                    #       f"Received: {self._total_samples_received}")

            # # Log underruns but not too frequently
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
                    # print(f"AudioPlayer: End marker received, "
                    #       f"buffer has {len(self._audio_buffer)} samples")
                    # print(f"AudioPlayer: Total received: {self._total_samples_received}, "
                    #       f"Total played: {self._total_samples_played}")
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
                text = await google_asr(track, stop_event)
                
                if not text or text.strip() == "":
                    print(f"{pc_id}: Empty ASR result")
                    await audio_queue.put(None)
                    return
                
                print(f"{pc_id}: ASR result: '{text}'")
                
                # Start pipeline tasks of LLM and TTS
                pc._llm_task = asyncio.create_task(
                    llm_generator(text, llm_queue, llm_client, SYSTEM_PROMPT, MAX_TOKENS)
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
        language_code=ASR_LANGUAGE,
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
                responses = google_asr_client.streaming_recognize(
                    google_asr_streaming_config, 
                    requests,
                )
                transcript = listen_print_loop(responses, audio_stream)
            output_queue.put(transcript)
        asr_thread = threading.Thread(target=run_google_stream, daemon=True)
        frame = await track.recv()
        # Once connection established and recieves the first dummy frame, start the ASR thread
        print('='*50)
        print("Start recording audio")
        print('='*50)
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

async def llm_generator(
    text: str, 
    output_queue: asyncio.Queue, 
    llm_client: object, 
    system_prompt=None, 
    max_tokens=512
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
    print(f"LLM: Generating response for: '{text}'")
    total_output = ""
    llm_start_time = time.time()
    try:
        # Stream the response
        chat_completion = llm_client.generate(
            users=[text],
            system=system_prompt,
            max_tokens=max_tokens,
        )
        init = True
        # Buffer for combining small chunks
        buffer = ""
        for chunk in chat_completion:
            response = chunk.choices[0].delta.content
            
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

async def online_tts(input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    """
    Optimized TTS that truly streams audio as it's generated.
    
    Key optimizations:
    1. True streaming - output audio as soon as chunks arrive
    2. Minimal buffering - reduce latency
    3. Overlap processing - start next TTS while current one streams
    4. Adaptive sentence splitting for better responsiveness
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
                        samples = await process_streaming_tts(
                            text_buffer.strip(), output_queue, sentence_count
                        )
                        total_samples_generated += samples
                    break
                
                print(f"TTS: Received text chunk: '{text_chunk[:50]}...'")
                text_buffer += text_chunk
                
                # Use more aggressive sentence splitting for lower latency
                sentences, text_buffer = extract_sentences_aggressive(text_buffer)
                
                for sentence in sentences:
                    if sentence.strip():
                        sentence_count += 1
                        
                        # Wait for previous TTS to finish before starting new one
                        # This prevents overwhelming the API while maintaining order
                        if current_tts_task and not current_tts_task.done():
                            await current_tts_task
                        
                        # Start new streaming TTS
                        current_tts_task = asyncio.create_task(
                            process_streaming_tts(sentence.strip(), output_queue, sentence_count)
                        )
                        
                        # For the first sentence, wait a bit to get audio flowing
                        if sentence_count == 1:
                            # Give it a small head start but don't wait for completion
                            try:
                                await asyncio.wait_for(asyncio.shield(current_tts_task), timeout=0.5)
                            except asyncio.TimeoutError:
                                pass  # Continue processing, audio will stream in background
                        
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


async def process_streaming_tts(sentence: str, output_queue: asyncio.Queue, sentence_id: int) -> int:
    """
    Process TTS with true streaming - output audio chunks as they arrive.
    
    This eliminates the 5-second delay by streaming audio immediately.
    """
    if not sentence.strip():
        return 0
    
    start_time = time.time()
    total_samples = 0
    first_chunk_time = None
    
    try:
        print(f"TTS: Starting streaming TTS for sentence {sentence_id}: '{sentence[:50]}...'")
        
        # Create the streaming request
        def request_generator():
            yield google_tts_config_request
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=sentence)
            )
        
        # Use a thread-safe queue to get audio chunks as they arrive
        audio_chunk_queue = queue.Queue()
        error_container = [None]
        
        def run_streaming_tts():
            """Run TTS in thread and put chunks in queue as they arrive"""
            try:
                streaming_responses = google_tts_client.streaming_synthesize(request_generator())
                
                # Process each response immediately (TRUE STREAMING)
                for response in streaming_responses:
                    if response.audio_content:
                        audio_chunk_queue.put(('audio', response.audio_content))
                
                audio_chunk_queue.put(('done', None))
                
            except Exception as e:
                error_container[0] = e
                audio_chunk_queue.put(('error', str(e)))
        
        # Start TTS in background thread
        tts_thread = threading.Thread(target=run_streaming_tts, daemon=True)
        tts_thread.start()
        
        # Process audio chunks as they arrive
        while True:
            try:
                # Wait for next audio chunk (with timeout)
                chunk_type, chunk_data = audio_chunk_queue.get(timeout=10.0)
                
                if chunk_type == 'error':
                    print(f"TTS: Error in sentence {sentence_id}: {chunk_data}")
                    break
                elif chunk_type == 'done':
                    print(f"TTS: Streaming complete for sentence {sentence_id}")
                    break
                elif chunk_type == 'audio':
                    # Process and send audio chunk immediately
                    audio_np = np.frombuffer(chunk_data, dtype=np.int16)
                    
                    if len(audio_np) > 0:
                        # Apply improvements
                        audio_np = apply_audio_improvements(audio_np)
                        total_samples += len(audio_np)
                        
                        # Send immediately in optimized chunks
                        await send_audio_immediately(audio_np, output_queue)
                        
                        # Track first chunk timing
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            latency = first_chunk_time - start_time
                            print(f"TTS: First audio chunk for sentence {sentence_id} in {latency:.3f}s")
                
            except queue.Empty:
                print(f"TTS: Timeout waiting for audio chunk in sentence {sentence_id}")
                break
        
        processing_time = time.time() - start_time
        print(f"TTS: Sentence {sentence_id} completed in {processing_time:.2f}s, {total_samples} samples")
        
        return total_samples
        
    except Exception as e:
        print(f"TTS: Error in streaming TTS for sentence {sentence_id}: {e}")
        import traceback
        traceback.print_exc()
        return 0


async def send_audio_immediately(audio_np: np.ndarray, output_queue: asyncio.Queue):
    """Send audio chunks immediately without waiting for complete sentence."""
    if len(audio_np) == 0:
        return
    
    # Use smaller chunks for lower latency
    chunk_size = SAMPLES_PER_FRAME * 5  # 50ms chunks instead of 100ms
    
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
    
    import re
    
    sentences = []
    
    # Split on various punctuation for more responsive streaming
    # Include commas, semicolons, and other natural pause points
    parts = re.split(r'([.!?。！？，,；;：:]+)', text_buffer)
    
    current_sentence = ""
    i = 0
    
    while i < len(parts) - 1:
        current_sentence += parts[i]
        
        if i + 1 < len(parts) and re.match(r'[.!?。！？，,；;：:]+', parts[i + 1]):
            current_sentence += parts[i + 1]
            
            # Only create sentence if it's long enough to be worth processing
            if len(current_sentence.strip()) > 10:  # Minimum 10 characters
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