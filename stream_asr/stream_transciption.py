# transcribe the voice into text without streaming

from ollama import chat
import asyncio
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
import yaml
import pywhispercpp.constants
from pywhispercpp.model import Model
import pywhispercpp
from pydub import AudioSegment
import os
from pathlib import Path
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import queue
from multiprocessing import Process, Queue, Value
from ctypes import c_bool

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
asr_model = config.get('model', 'tiny')

SAMPLE_RATE = 16000 
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01 
SILENCE_DURATION = 1  # seconds
MAX_SILENT_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)

silent_chunks = 0
valid_chunks = 1
buffer_chunks = []

def rms_energy(data):
    return np.sqrt(np.mean(data**2))

def record_audio(audio_queue, stop_signal):
    start_signal = False
    def audio_callback(indata, frames, time, status):
        nonlocal start_signal
        audio_chunk = indata[:, 0]
        energy = rms_energy(audio_chunk)
        if energy < SILENCE_THRESHOLD:
            if start_signal:
                silent_chunks += 1
        else:
            start_signal = True
            silent_chunks = 0
        audio_queue.put(audio_chunk.flatten())
    
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=CHUNK_SIZE,):
        try:
            while True:
                if not stop_signal.value:
                    sd.sleep(1)
        except KeyboardInterrupt:
            print("Recording stopped by user.")

def transcribe_audio(model, audio_queue, stop_signal):
    model = Model(asr_model)
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=1)
        except queue.Empty:
            if not stop_signal.value:
                break
            continue
        result = model.transcribe(audio_chunk, language='zh')
        
        print("Transcribed text:", result[0].text.strip())
    
#     nonlocal silent_chunks, valid_chunks, start_recording, texts, buffer_chunks
#     audio_chunk = indata[:, 0]
#     energy = rms_energy(audio_chunk)
#     print(energy)
#     if energy < SILENCE_THRESHOLD:
#         if start_recording:
#             silent_chunks += 1
#     else:
#         start_recording = True
#         silent_chunks = 0
        
#     if start_recording:
#         buffer_chunks.append(audio_chunk)
#         if len(buffer_chunks) >= CHUNKS_PER_TRANSCRIBE * valid_chunks:
#             curr_chunks = np.concatenate(buffer_chunks[-CHUNKS_PER_TRANSCRIBE:])
#             print(len(curr_chunks), curr_chunks.shape)
#             print(silent_chunks, valid_chunks)
#             # write(f"test_audio/recording_test_{valid_chunks}.wav", SAMPLE_RATE, (curr_chunks * 32767).astype(np.int16))
#             transcribed_text = model.transcribe((curr_chunks * 32767).astype(np.int16), language='zh')
#             print(len(transcribed_text), transcribed_text)
#             for chunk in transcribed_text:
#                 print("实时识别:", chunk.text.strip())
#                 texts += chunk.text.strip() + ' '
#             valid_chunks += 1

#     if silent_chunks > MAX_SILENT_CHUNKS:
#         print("====================Stop recording====================")
#         raise sd.CallbackStop()

# print("====================Start recording====================")
# with sd.InputStream(
#     samplerate=SAMPLE_RATE,
#     channels=1,
#     dtype='float32',
#     callback=audio_callback,
#     blocksize=CHUNK_SIZE,
# ):
#     sd.sleep(int(10 * 1000))  # Max 10 seconds, or until silence

if __name__ == "__main__":
    audio_queue = Queue()
    stop_signal = Value(c_bool, False)
    recorder = Process(target=record_audio, args=(audio_queue, stop_signal))
    transcriber = Process(target=transcribe_audio, args=(asr_model, audio_queue, stop_signal))
    recorder.start()
    transcriber.start()
    recorder.join()
    transcriber.join()

