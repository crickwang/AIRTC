from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import yaml
import re

with open('tts_test.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
text = config.get('test_text', 'large_text').replace(' ', '').strip()
voice = config.get('kokoro_voice', 'zm_yunjian')
language = config.get('kokoro_language', 'z')

reexp = r'(?<=[.!?。])\s*'
    
def main() -> float:
    """
    Main function to run the TTS pipeline using Kokoro.
    It initializes the pipeline, processes the text, and plays the audio.
    kokoro reference: [https://github.com/hexgrad/kokoro]
    Returns:
        float: The total time taken for the TTS process.
    """
    audio_queue = queue.Queue()
    start = time.time()
    print('===================开始计时=====================')

    pipeline = KPipeline(lang_code=language) # 查看kokoro的文档了解更多语言选项

    play_thread = threading.Thread(target=second_worker, args=(audio_queue,))
    play_thread.start()

    chunks = re.split(reexp, text)

    for chunk in chunks:
        if not chunk.strip():
            continue
        generator = pipeline(chunk, voice=voice)
        model_time = time.time()
        print(f'================训练模型用时: {(model_time - start):.2f} seconds================')
        start = model_time
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            audio_queue.put(audio)

    audio_queue.put(None)
    play_thread.join()
    
    return start

# 第二个进程
def second_worker(audio_queue):
    """    
    Function to play audio from the audio queue.
    It continuously retrieves audio data from the queue and plays it using sounddevice.
    Args:
        audio_queue (queue.Queue): The queue containing audio data to be played.
    Returns:
        None
    """
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        sd.play(audio, samplerate=24000)
        sd.wait()
        
if __name__ == "__main__":
    print('===================开始kokoro TTS=====================')
    start = main()
    print('===================结束计时=====================')
    print(f'总用时: {time.time() - start:.2f} seconds')