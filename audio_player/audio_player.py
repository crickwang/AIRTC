from aiortc import AudioStreamTrack
import numpy as np
import asyncio
from collections import deque
import time
from av import AudioFrame
from fractions import Fraction

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
    def __init__(self, audio_queue: asyncio.Queue, sample_rate: int, samples_per_frame: int, format: str, layout: str) -> None:
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self._timestamp = 0
        self.samples_per_frame = samples_per_frame
        self.format = format
        self.layout = layout
        
        # buffer management
        self._audio_buffer = deque()
        self._max_buffer_ms = 6000000  # Maximum 6000000ms of audio buffered
        self._max_buffer_samples = (self._max_buffer_ms * sample_rate) // 1000
        
        # State tracking
        self._is_playing = False
        self._interrupt_requested = False
        self._frames_sent = 0
        
        # Timing control
        self._last_frame_time = time.time()
        self._frame_interval = self.samples_per_frame / self.sample_rate

        print(f"AudioPlayer: Initialized")

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
                self._is_playing = False
        
        # Create and return frame
        frame = AudioFrame.from_ndarray(audio_array, format=self.format, layout=self.layout)
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
            audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=3)
            
            if self._interrupt_requested:
                self._clear_buffers()
                self._interrupt_requested = False
                self._is_playing = False
            
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
        except asyncio.TimeoutError:
            pass
        except asyncio.QueueEmpty:
            pass
    
    def request_interrupt(self):
        """Request immediate interruption of audio playback."""
        self._interrupt_requested = True
        #print(f"AudioPlayer: Interrupt requested")
    
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
        sample_rate: int = 16000,
        samples_per_frame: int = 1024,
        format: str = "s16",
        layout: str = "mono"
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
        self.samples_per_frame = samples_per_frame
        self.foramt = format
        self.layout = layout

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
            frame = AudioFrame.from_ndarray(audio_array, format=self.format, layout=self.layout)
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
        frame = AudioFrame.from_ndarray(silence, format=self.format, layout='mono')
        frame.sample_rate = self.sample_rate
        frame.time_base = Fraction(1, self.sample_rate)
        frame.pts = self._timestamp
        self._timestamp += self.samples_per_frame
        return frame