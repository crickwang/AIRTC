# transcribe the voice into text without streaming

import yaml
import pywhispercpp.constants
from pywhispercpp.model import Model
import pywhispercpp
from pydub import AudioSegment
import os
import sounddevice as sd
import numpy as np
import time
from multiprocessing import Queue, Value, Pool
from ctypes import c_bool, c_wchar_p

SAMPLE_RATE = 16000 
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.0215 # 0.01: sensitive, 0.02: normal voice, 0.05: shouting
SILENCE_DURATION = 2  # seconds
MAX_SILENT_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)
MAX_RECORDING_DURATION = 30  # seconds
BUFFER_DURATION = 2  # seconds
CHUNKS_PER_BUFFER = int(BUFFER_DURATION / CHUNK_DURATION)

def initializer(asr_model):
    global model
    model = Model(asr_model)

def rms_energy(data):
    return np.sqrt(np.mean(data**2))

def transcribe_worker(audio_data, language='zh'):
    print("Transcribing audio...")
    global model
    result = model.transcribe((audio_data * 32767).astype(np.int16), language=language)
    text = result[0].text.strip()
    print(f'=========================text: {text}=============================')
    return text

def record_and_stream(pool, stop_signal):
    buffer_chunks = []
    start_signal = False
    silent_chunks = 0
    start_time = time.time()
    
    def handle_callback(result):  
        print(f'Result from transcription: {result}')
        text_queue.put(result)
    
    def audio_callback(indata, frames, time, status):
        nonlocal buffer_chunks, silent_chunks, start_signal, start_time
        audio_chunk = indata[:, 0]
        energy = rms_energy(audio_chunk)
        #print('energy level:', energy)
        if energy < SILENCE_THRESHOLD:
            if start_signal:
                silent_chunks += 1
        else:
            start_signal = True
            silent_chunks = 0
        if start_signal:
            buffer_chunks.append(audio_chunk)
            if len(buffer_chunks) >= CHUNKS_PER_BUFFER:
                audio_data = np.concatenate(buffer_chunks)
                pool.apply_async(transcribe_worker, args=(audio_data,), callback=handle_callback)
                buffer_chunks = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=CHUNK_SIZE,
    ):
        while not stop_signal.value:
            if silent_chunks > MAX_SILENT_CHUNKS or time.time() - start_time > MAX_RECORDING_DURATION:
                print("====================Stop recording====================")
                stop_signal.value = True
                break

if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    asr_model = config.get('model', 'tiny')
    #model = Model(model=asr_model)
    model = None
    print('1')
    stop_signal = Value(c_bool, False)
    text_queue = Queue()
    text = ''
    print('2')
    
    print("====================Start speaking====================")
    with Pool(processes=3, initializer=initializer, initargs=(asr_model,)) as pool:
        try:
            record_and_stream(pool, stop_signal)
        except KeyboardInterrupt:
            stop_signal.value = True
            print("Recording stopped by user.")
    pool.close()
    pool.join()
    while not text_queue.empty():
        text += text_queue.get()
    print(text)

