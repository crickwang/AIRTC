# transcribe the voice with no streaming

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pywhispercpp.model import Model
import yaml
import os
import time

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
MODEL = config.get('whisper_model', 'tiny')
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_file = config.get('output_wav', 'test.wav')
output_file = os.path.join(PARENT_DIR, 'audio', output_file)

SAMPLE_RATE = 16000 
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01 
SILENCE_DURATION = 2  # seconds
MAX_SILENT_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)


# calculate the RMS energy of audio
model = Model(MODEL, models_dir=os.path.join(ROOT, "model"))
def rms_energy(data):
    return np.sqrt(np.mean(data**2))


print("====================Start speaking====================")
recorded_audio = []
silent_chunks = 0
start_recording = False

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
    while True:
        chunk, _ = stream.read(CHUNK_SIZE)
        energy = rms_energy(chunk)
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
write(output_file, SAMPLE_RATE, (recorded_audio * 32767).astype(np.int16))
start_time = time.time()
text = model.transcribe((recorded_audio * 32767).astype(np.int16), language='zh')
print(f"Transcription took {time.time() - start_time:.2f} seconds")
print('tot text', text)
print("Transcribed text:", text[0].text.strip())
    


