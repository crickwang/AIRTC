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
from openai import OpenAI
from dotenv import load_dotenv

SAMPLE_RATE = 16000 
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.01 # 0.05 for normal volume
SILENCE_DURATION = 2  # seconds
MAX_SILENT_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)
CHUNKS_PER_TRANSCRIBE = int(2 / CHUNK_DURATION) # Transcribe every 2 seconds

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

llm_model = config.get('llm_model', 'deepseek-r1:1.5b')
asr_model = config.get('asr_model', 'tiny')
m4a_file = config.get('mandarin_m4a', 'audio/mandarin.m4a')
voice = config.get('tts_voice', 'en-GB-SoniaNeural')
output_file = config.get('tts_output', 'audio/tts_output.mp3')
srt_file = config.get('tts_subtitle', 'audio/tts_output.srt')

def asr_recording(m4a_file, asr_model):
    wav_file = 'audio/temp.wav'
    audio = AudioSegment.from_file(m4a_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_file, format="wav")

    model = Model(asr_model)
    chunks = model.transcribe(wav_file, language='zh')
    texts = ''
    for chunk in chunks:
        texts += chunk.text.strip()
    print(f'asr outputs: {texts}')
    return texts

# def asr_live(asr_model):
#     silent_chunks = 0
#     valid_chunks = 1
#     start_recording = False
#     texts = ''
#     buffer_chunks = []
#     model = Model(asr_model)

#     def rms_energy(data):
#         return np.sqrt(np.mean(data**2))

#     def audio_callback(indata, frames, time, status):
#         nonlocal silent_chunks, valid_chunks, start_recording, texts, buffer_chunks
#         audio_chunk = indata[:, 0]
#         energy = rms_energy(audio_chunk)
#         print(energy)
#         if energy < SILENCE_THRESHOLD:
#             if start_recording:
#                 silent_chunks += 1
#         else:
#             start_recording = True
#             silent_chunks = 0
            
#         if start_recording:
#             buffer_chunks.append(audio_chunk)
#             if len(buffer_chunks) >= CHUNKS_PER_TRANSCRIBE * valid_chunks:
#                 curr_chunks = np.concatenate(buffer_chunks[-CHUNKS_PER_TRANSCRIBE:])
#                 print(len(curr_chunks), curr_chunks.shape)
#                 print(silent_chunks, valid_chunks)
#                 # write(f"test_audio/recording_test_{valid_chunks}.wav", SAMPLE_RATE, (curr_chunks * 32767).astype(np.int16))
#                 transcribed_text = model.transcribe((curr_chunks * 32767).astype(np.int16), language='zh')
#                 print(len(transcribed_text), transcribed_text)
#                 for chunk in transcribed_text:
#                     print("实时识别:", chunk.text.strip())
#                     texts += chunk.text.strip() + ' '
#                 valid_chunks += 1

#         if silent_chunks > MAX_SILENT_CHUNKS:
#             print("====================Stop recording====================")
#             raise sd.CallbackStop()

#     print("====================Start recording====================")
#     with sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         channels=1,
#         dtype='float32',
#         callback=audio_callback,
#         blocksize=CHUNK_SIZE,
#     ):
#         sd.sleep(int(10 * 1000))  # Max 10 seconds, or until silence
#     print(texts)
#     return texts

def llm_local(texts, llm_model):
    messages = [
        {'role': 'system', 'content': '使用中文回答问题。不要生成任何图像符号,如emoji等不属于文字的内容。'},
        {'role': 'user', 'content': texts}]
    
    stream = chat(
        model=llm_model,
        messages=messages,
        stream=True,
    )

    contents = ''
    for chunk in stream:
        content = chunk['message']['content']
        contents += content
    print(f'llm outputs: {contents}')
    return contents

def llm_online(texts):
    load_dotenv()
    client = OpenAI(
        api_key=os.environ.get("API_KEY"),
        base_url=os.environ.get("BASE_URL"),
    )

    messages = [
        {'role': 'system', 'content': '你是一只可爱的猫娘，回答每句话的时候最后都会加一句喵'},
        {'role': 'user', 'content': texts}
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=os.environ.get("MODEL"),
        stream=True,
        max_completion_tokens=1024,
    )

    contents = ''
    for chunk in chat_completion:
        contents += chunk.choices[0].delta.content or ""
    
    print(f'llm outputs: {contents}')
    return contents

async def tts(contents) -> None:
    com = communicate.Communicate(
        text=contents, voice=voice)
    sub = submaker.SubMaker()
    with open(output_file, "wb") as file:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                sub.feed(chunk)

    with open(srt_file, "w", encoding="utf-8") as file:
        file.write(sub.get_srt())

async def amain() -> None:
    start = time.time()
    texts = asr_recording(m4a_file, asr_model)
    #texts = asr_live(asr_model)
    asr_time = time.time()
    print(f"ASR time: {asr_time - start:.2f} seconds")
    start = asr_time
    #contents = llm_local(texts, llm_model)
    contents = llm_online(texts)
    llm_time = time.time()
    print(f"LLM time: {llm_time - start:.2f} seconds")
    start_time = llm_time
    await tts(contents)
    tts_time = time.time()
    print(f"TTS time: {tts_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(amain())
