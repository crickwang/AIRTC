import numpy as np
from abc import ABC, abstractmethod
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
    
@register.add_model("vad", "multiFrame")
class MultiFrameVAD(VAD):
    """
    Voice Activity Detection with multiple frames. More robust against noise.
    Not advised to use this VAD, as it may swallow first several frames. 
    You may adjust you own VAD if you want.
    """
    def __init__(self, threshold, speech_frames_required=3):
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
        Returns:
            VAD: The created VAD instance.
        """
        try:
            vad = register.get_model("vad", algorithm)
            return vad(**kwargs)
        except Exception as e:
            print(f"Error creating VAD instance: {e}")
            return None