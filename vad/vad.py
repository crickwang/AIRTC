from abc import ABC, abstractmethod

import numpy as np

from register import register


class VAD(ABC):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold

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

    def drain_preroll(self) -> list:
        """
        Return the frames buffered just before speech was confirmed, and clear them.

        A VAD that only confirms speech after several consecutive frames has, by
        definition, already consumed the start of the utterance by the time it
        returns True. Callers should push these frames into the ASR queue ahead of
        the current frame so the first syllable isn't clipped. VADs with no
        look-back buffer return an empty list.
        Returns:
            list[np.ndarray]: Buffered frames, oldest first. Excludes the frame
                              that was just passed to is_speech().
        """
        return []

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
                 preroll_frames=10):
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
        """
        self.threshold = threshold
        self.speech_frames_required = speech_frames_required
        self.silence_frames_required = silence_frames_required
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_currently_speaking = False
        # store the previous frames and prevent them from being swallowed if they
        # are meaningful speeches. Sized to cover at least the onset window, or the
        # very frames that triggered detection would themselves be lost.
        self.frames = [None for i in range(max(preroll_frames, speech_frames_required))]

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

    def drain_preroll(self) -> list:
        """
        Return the buffered frames leading up to the current one, and clear the buffer.
        See VAD.drain_preroll. The last slot holds the frame just passed to is_speech(),
        which the caller already has, so it is excluded here to avoid queueing it twice.
        Returns:
            list[np.ndarray]: Buffered frames, oldest first.
        """
        preroll = [frame for frame in self.frames[:-1] if frame is not None]
        self.frames = [None for i in range(len(self.frames))]
        return preroll

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
    """
    def __init__(self, threshold):
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
