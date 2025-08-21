from ollama import chat
import asyncio
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
import yaml
from pywhispercpp.model import Model
from pydub import AudioSegment
import os
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
MAX_TOKEN = 1024

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

llm_model = config.get('llm_model', 'deepseek-r1:1.5b')
asr_model = config.get('asr_model', 'tiny')
m4a_file = config.get('mandarin_m4a', 'audio/mandarin.m4a')
voice = config.get('tts_voice', 'en-GB-SoniaNeural')
output_file = config.get('tts_output', 'audio/tts_output.mp3')
srt_file = config.get('tts_subtitle', 'audio/tts_output.srt')
llm_online_model = config.get('llm_online_model', 'ernie-4.5-0.3b')
system_prompt = config.get('system_prompt')

def asr_recording(m4a_file: str, asr_model: str) -> str:
    """
    Transcribe audio from a file using Whisper ASR model.
    Args:
        m4a_file (str): Path to the input audio file in m4a format.
        asr_model (str): The name of the ASR model to use for transcription.
    Returns:
        str: The transcribed text.
    """
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

def llm_local(texts: str, llm_model: str) -> str:
    """
    Transcribe text using a local LLM model.
    Args:
        texts (str): The input text to transcribe.
        llm_model (str): The name of the LLM model to use for transcription.
    Returns:
        str: The transcribed text.
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': texts}
    ]

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

def llm_online(texts: str, max_token: int = MAX_TOKEN) -> str:
    """
    Transcribe text using an online LLM model.
    Please refer to [https://ai.baidu.com/ai-doc/AISTUDIO] for more details.
    Use [https://aistudio.baidu.com/account/accessToken] to get the access token.
    please configure the .env file with your API key and base URL.
    Args:
        texts (str): The input text to transcribe.
        max_token (int, optional): The maximum number of tokens for the LLM response.
            Defaults to MAX_TOKEN.
    Returns:
        str: The transcribed text.
    """
    load_dotenv()
    client = OpenAI(
        api_key=os.environ.get("BAIDU_AISTUDIO_API_KEY"),
        base_url=os.environ.get("BAIDU_AISTUDIO_BASE_URL"),
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': texts}
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=llm_online_model,
        stream=True,
        max_completion_tokens=max_token,
    )

    contents = ''
    for chunk in chat_completion:
        contents += chunk.choices[0].delta.content or ""
    
    print(f'llm outputs: {contents}')
    return contents

async def tts(contents: str) -> None:
    """
    Convert text to speech using Edge TTS and save the output to a file.
    Args:
        contents (str): The text to convert to speech.
    Returns:
        None
    """
    com = communicate.Communicate(text=contents, voice=voice)
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
