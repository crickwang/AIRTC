
import numpy as np
from pywhispercpp.model import Model
import yaml
import os
import time

SAMPLE_RATE = 16000 
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01 
SILENCE_DURATION = 2  # seconds
MAX_SILENT_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("====================Start speaking====================")
recorded_audio = []
silent_chunks = 0
start_recording = False

class WhisperTranscription:
    """
    A class to handle Whisper transcription.
    """
    def __init__(self, model: str) -> None:
        """
        Initialize the WhisperTranscription class.
        Args:
            model (str): The name of the Whisper model to use for transcription.
            The model should be available in the 'model' directory.
        Returns:
            None
        """
        self.model = model

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data to text.
        Args:
            audio_data (np.ndarray): The audio data to transcribe.
        Returns:
            str: The transcribed text.
        """
        return self.model.transcribe(audio_data, language='zh')

if __name__ == "__main__":
    from scipy.io.wavfile import write
    import sounddevice as sd
    
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        asr_model = config.get('whisper_model', 'tiny')
        output_wav = config.get('output_wav', 'test.wav')
        print(f"Using ASR model: {asr_model}")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            energy = np.sqrt(np.mean(chunk**2))
            print(energy)
            recorded_audio.append(chunk)
            print(len(recorded_audio))

            if energy < SILENCE_THRESHOLD:
                if start_recording:
                    silent_chunks += 1
            else:
                start_recording = True
                silent_chunks = 0

            if silent_chunks > MAX_SILENT_CHUNKS:
                print("====================Stop recording====================")
                break

    recorded_audio = np.concatenate(recorded_audio, axis=0)
    write(output_wav, SAMPLE_RATE, (recorded_audio * 32767).astype(np.int16))
    start_time = time.time()

    whisper_transcription = WhisperTranscription(asr_model)
    text = whisper_transcription.transcribe((recorded_audio * 32767).astype(np.int16))
    print(f"Transcription took {time.time() - start_time:.2f} seconds")
    print('tot text', text)
    print("Transcribed text:", text[0].text.strip())