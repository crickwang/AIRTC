import pywhispercpp.constants
from pywhispercpp.model import Model
import yaml
import pywhispercpp
from pydub import AudioSegment
import os
from pathlib import Path
import time

with open('asr_test.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
model_name = config.get('model', 'tiny')
mp3_file = config.get('mp3_file')
wav_file = config.get('wav_file')

if wav_file is None:
    if mp3_file is None:
        raise ValueError("Please provide either an mp3_file or a wav_file.")
    else:
        wav_file = 'temp.wav'
        audio = AudioSegment.from_mp3(mp3_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_file, format="wav")

model = Model(model=model_name)
chunks = model.transcribe(wav_file, language='zh')
for chunk in chunks:
    print(chunk.text.strip())
    
# print(pywhispercpp.constants.AVAILABLE_MODELS)
    
# start = time.time()
# chunks1 = model.transcribe(wav_file, language='zh')
# end = time.time()
# print(f"Time taken for second transcription: {end - start} seconds")
# start = end
# chunks2 = model.transcribe(wav_file, language='zh')
# end = time.time()
# print(f"Time taken for third transcription: {end - start} seconds")
# start = end

