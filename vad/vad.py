from abc import ABC, abstractmethod

import numpy as np

from register import register


class VAD(ABC):
    def __init__(self, threshold, preroll_frames=0, **kwargs):
        """
        Args:
            threshold: Energy threshold, for the VADs that work on raw energy.
                       Model-based VADs ignore it.
            preroll_frames: Size of the look-back ring buffer. 0 disables it, which
                            makes drain_preroll() return an empty list.
        """
        self.threshold = threshold
        # Ring buffer of the most recent frames. A VAD only confirms speech after
        # the utterance has already started, so these are replayed into the ASR
        # queue at onset rather than dropped.
        self.frames = [None for i in range(preroll_frames)]

    @abstractmethod
    def is_speech(self, frame: np.ndarray, **kwargs) -> bool:
        """
        Detect if frame contains speech.
        Args:
            frame (np.ndarray): The audio frame to analyze.
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        pass

    def populate(self, frame: np.ndarray) -> None:
        """
        Push a frame into the look-back buffer, dropping the oldest one.
        Args:
            frame (np.ndarray): The audio frame to buffer.
        Returns:
            None
        """
        if not self.frames:
            return
        self.frames.pop(0)
        self.frames.append(frame)

    def drain_preroll(self) -> list:
        """
        Return the frames buffered just before speech was confirmed, and clear them.

        A VAD that only confirms speech after several frames has, by definition,
        already consumed the start of the utterance by the time it returns True.
        Callers should push these frames into the ASR queue ahead of the current
        frame so the first syllable isn't clipped. The newest slot holds the frame
        just passed to is_speech(), which the caller already has, so it is excluded
        to avoid queueing it twice. VADs with no look-back buffer return [].
        Returns:
            list[np.ndarray]: Buffered frames, oldest first.
        """
        preroll = [frame for frame in self.frames[:-1] if frame is not None]
        self.frames = [None for i in range(len(self.frames))]
        return preroll

@register.add_model("vad", "multiFrame")
class MultiFrameVAD(VAD):
    """
    Voice Activity Detection with multiple frames. More robust against noise.
    Requires speech_frames_required consecutive loud frames to start and
    silence_frames_required consecutive quiet frames to stop. The frames leading up
    to detection are kept and handed back by drain_preroll(), so callers can replay
    them into the ASR queue instead of swallowing the start of the utterance.
    You may adjust you own VAD if you want.
    """
    def __init__(self, threshold, speech_frames_required=3, silence_frames_required=10,
                 preroll_frames=10, input_sample_rate=None):
        """
        Initialize MultiFrameVAD instance
        Args:
            self: The instance of the class.
            threshold: The energy threshold for detecting speech.
                       In general, noise < 1000 while speech > 2000.
                       Maybe adjusted based on the noise level of the environment.
            speech_frames_required: The number of consecutive frames required to confirm speech.
            silence_frames_required: The number of consecutive quiet frames required to
                       confirm speech has ended (release hangover). At TIME_PER_CHUNK=10ms
                       the default is 100ms, enough to ride over the energy dips between
                       syllables without delaying end-of-turn much. The ASR loops add their
                       own max_silence_chunk (500ms) on top of this before closing a turn.
            preroll_frames: How many recent frames to keep for drain_preroll(), i.e. how
                       much audio from before the detection point is replayed into the
                       ASR queue. Default 100ms at TIME_PER_CHUNK=10ms.
            input_sample_rate: Unused — energy detection is rate-agnostic. Accepted so
                       the pipeline can pass its rate to any VAD uniformly.
        """
        # The buffer must cover at least the onset window, or the very frames that
        # triggered detection would themselves be lost.
        super().__init__(threshold, preroll_frames=max(preroll_frames, speech_frames_required))
        self.speech_frames_required = speech_frames_required
        self.silence_frames_required = silence_frames_required
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_currently_speaking = False

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

        # Use hysteresis for more stable detection: speech starts only after
        # speech_frames_required consecutive loud frames and ends only after
        # silence_frames_required consecutive quiet frames.
        self.populate(frame)
        if rms_energy > self.threshold:
            # A loud frame cancels any release in progress.
            self.silence_frame_count = 0
            if not self.is_currently_speaking:
                self.speech_frame_count += 1
                if self.speech_frame_count >= self.speech_frames_required:
                    print(f"VAD: Speech detected (energy: {rms_energy:.1f})")
                    self.is_currently_speaking = True
                    self.speech_frame_count = 0
        else:
            # Onset must be consecutive, so any quiet frame restarts the count.
            self.speech_frame_count = 0
            if self.is_currently_speaking:
                self.silence_frame_count += 1
                if self.silence_frame_count >= self.silence_frames_required:
                    print(f"VAD: Speech ended (energy: {rms_energy:.1f})")
                    self.is_currently_speaking = False
                    self.silence_frame_count = 0
        return self.is_currently_speaking

@register.add_model("vad", "simple")
class SimpleVAD(VAD):
    """
    A simple Voice Activity Detection (VAD) class
    that only measures the energy of a single audio frame.
    It fires on the first frame that crosses the threshold, so it reacts fast but
    accepts isolated noise (a click or a door slam) as speech. The frames leading up
    to detection are buffered and handed back by drain_preroll(), since the quiet
    leading edge of a word sits below the threshold by definition.
    """
    def __init__(self, threshold, preroll_frames=10, input_sample_rate=None):
        """
        Initialize SimpleVAD instance.
        Args:
            self: The instance of the class.
            threshold: The energy threshold for detecting speech.
                       In general, noise < 2000 while speech > 20000.
                       Maybe adjusted based on the noise level of the environment.
            preroll_frames: How many recent frames to keep for drain_preroll(). This
                       VAD triggers on the frame that crosses the threshold, so without
                       a look-back the softer onset before it is lost. Default 100ms at
                       TIME_PER_CHUNK=10ms.
            input_sample_rate: Unused — energy detection is rate-agnostic. Accepted so
                       the pipeline can pass its rate to any VAD uniformly.
        """
        super().__init__(threshold, preroll_frames=preroll_frames)

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
        self.populate(frame)
        return bool(rms_energy > self.threshold)

@register.add_model("vad", "fsmn")
class FSMNVAD(VAD):
    """
    Neural VAD backed by FunASR's FSMN-VAD (iic/speech_fsmn_vad_zh-cn-16k-common-pytorch).

    Unlike the energy VADs this classifies spectral features, so it separates speech
    from door slams, keyboard clicks and steady hum rather than just measuring
    loudness, and needs no threshold tuning per microphone or room.

    funasr is already a project dependency, so this adds no new package — but the
    model weights are downloaded from ModelScope on first use and cached under
    ~/.cache/modelscope. That download must succeed once before this VAD can run.

    Three properties of the model shape the integration:
      - It is a 16kHz model. Frames arriving at another rate are resampled here.
      - It works on chunks, not single frames, so 10ms frames are accumulated to
        chunk_size_ms before each inference. is_speech() reports the last known
        state in between, so callers still get a per-frame answer.
      - It does its own end-pointing (max_end_silence_time), so the onset/release
        hysteresis that MultiFrameVAD implements by hand is built in here.
    """
    MODEL_SAMPLE_RATE = 16000

    # AutoModel load is slow and the weights are read-only, so instances share one.
    _shared_models = {}

    def __init__(self, threshold=None, input_sample_rate=16000, chunk_size_ms=60,
                 max_end_silence_ms=500, preroll_frames=20, device="cpu"):
        """
        Initialize FSMNVAD instance.
        Args:
            threshold: Unused — the model has its own learned decision. Accepted so the
                       factory can construct any VAD with the same kwargs.
            input_sample_rate: Sample rate of the frames passed to is_speech(), i.e.
                       whatever the pipeline's resampler produces. Audio is resampled
                       to MODEL_SAMPLE_RATE when these differ.
            chunk_size_ms: How much audio to accumulate per inference. Smaller reacts
                       sooner but runs the model more often; the model's internal frame
                       is 10ms, so this should be a multiple of that.
            max_end_silence_ms: Silence needed before the model closes a speech segment.
                       FunASR's own default is 800ms; 500ms is used here because the ASR
                       loops then add their own max_silence_chunk (500ms) on top before
                       ending a turn.
            preroll_frames: Size of the look-back buffer replayed at speech onset.
                       Wants to be larger than for the energy VADs, since detection can
                       lag by up to chunk_size_ms plus the model's own decision delay.
            device: Torch device for inference. CPU is fine — the model is tiny.
        """
        super().__init__(threshold, preroll_frames=preroll_frames)
        self.input_sample_rate = input_sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.max_end_silence_ms = max_end_silence_ms
        self.device = device
        self.is_currently_speaking = False

        # Frames arrive at TIME_PER_CHUNK; batch them up to chunk_size_ms of input audio.
        self.chunk_samples = int(input_sample_rate * chunk_size_ms / 1000)
        self.pending = []
        self.pending_samples = 0
        self.cache = {}

        self.model = self._load_model(max_end_silence_ms, device)
        self.resampler = self._build_resampler(input_sample_rate)

    @classmethod
    def _load_model(cls, max_end_silence_ms, device):
        """
        Return the shared AutoModel for these settings, loading it on first use.
        funasr and torch are imported lazily so this module stays importable (and
        testable) without the ML stack installed.
        """
        key = (max_end_silence_ms, device)
        if key not in cls._shared_models:
            from funasr import AutoModel
            cls._shared_models[key] = AutoModel(model="fsmn-vad",
                                                max_end_silence_time=max_end_silence_ms,
                                                device=device,
                                                disable_update=True,
                                                disable_pbar=True,
                                                disable_log=True,
                                                )
        return cls._shared_models[key]

    @classmethod
    def _build_resampler(cls, input_sample_rate):
        """
        Return a resampler to MODEL_SAMPLE_RATE, or None if the rate already matches.
        funasr would resample internally, but it rebuilds the resampler on every call;
        holding one here builds the filter kernel once.
        """
        if input_sample_rate == cls.MODEL_SAMPLE_RATE:
            return None
        import torchaudio
        return torchaudio.transforms.Resample(input_sample_rate, cls.MODEL_SAMPLE_RATE)

    def _run_model(self, chunk: np.ndarray) -> None:
        """
        Run one inference over the accumulated chunk and update the speech state.

        In streaming mode the model reports segment edges rather than a per-frame
        verdict: [beg, -1] opens a segment, [-1, end] closes one, [beg, end] is a
        segment that opened and closed inside this chunk, and [] means no change.
        Args:
            chunk (np.ndarray): int16 samples at input_sample_rate.
        Returns:
            None
        """
        # The frontend scales by 1<<15 before computing features, so it wants [-1, 1].
        audio = chunk.astype(np.float32) / 32768.0
        if self.resampler is not None:
            import torch
            audio = self.resampler(torch.from_numpy(audio)).numpy()

        result = self.model.generate(input=audio,
                                     cache=self.cache,
                                     is_final=False,
                                     chunk_size=self.chunk_size_ms,
                                     fs=self.MODEL_SAMPLE_RATE,
                                     )
        if not result:
            return
        for start_ms, end_ms in result[0].get("value") or []:
            # end_ms == -1 means the segment is still open, so speech is ongoing.
            # Anything else carries a real end timestamp and closes it.
            self.is_currently_speaking = end_ms == -1

    def is_speech(self, frame: np.ndarray) -> bool:
        """
        Detect if frame contains speech.

        The model runs once per chunk_size_ms rather than once per frame, so between
        inferences this reports the most recent known state.
        Args:
            frame (np.ndarray): The audio frame to analyze, int16 at input_sample_rate.
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        if len(frame) == 0:
            return self.is_currently_speaking

        self.populate(frame)
        self.pending.append(frame)
        self.pending_samples += len(frame)
        if self.pending_samples < self.chunk_samples:
            return self.is_currently_speaking

        chunk = np.concatenate(self.pending)
        self.pending = []
        self.pending_samples = 0
        self._run_model(chunk)
        return self.is_currently_speaking


class VADFactory:
    @staticmethod
    def create(algorithm: str, **kwargs) -> VAD:
        """
        Create a VAD instance.

        Raises rather than returning None: callers gate on `if not vad or
        vad.is_speech(frame)`, so a None VAD silently means "every frame is speech",
        which fires an interrupt and spins up an ASR session on every single frame.
        A misconfigured VAD should fail loudly instead.
        Raises:
            ValueError: If algorithm is not a registered VAD.
        Returns:
            VAD: The created VAD instance.
        """
        vad = register.get_model("vad", algorithm)
        if vad is None:
            raise ValueError(f"Unknown VAD algorithm: {algorithm!r}. "
                             f"Available: {sorted(register.vads)}")
        return vad(**kwargs)
