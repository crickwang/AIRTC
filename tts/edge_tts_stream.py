import asyncio
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
import pyaudio as pa
from pathlib import Path
from pydub import AudioSegment
import sounddevice as sd
import re
import io
import numpy as np
import yaml
import os

with open("tts/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE = config.get("edge_voice", "zh-CN-XiaoxiaoNeural")
_out_file = config.get("edge_output", "output.mp3")
_srt_file = config.get("edge_subtitle", "output.srt")
OUTPUT_FILE = os.path.join(PARENT_DIR, _out_file)
SRT_FILE = os.path.join(PARENT_DIR, _srt_file)

class EdgeTTS:
    def __init__(self, voice=VOICE, output_file=OUTPUT_FILE, srt_file=SRT_FILE):
        self.voice = voice
        if output_file is not None:
            self.output_file = Path(output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.open("w").close()
        else:
            self.output_file = None
        if srt_file is not None:
            self.srt_file = Path(srt_file)
            self.srt_file.parent.mkdir(parents=True, exist_ok=True)
            self.srt_file.open("w").close()
        else:
            self.srt_file = None

    async def process_audio(self, text) -> None:
        chunks = []
        com = communicate.Communicate(text=text, voice=self.voice)
        sub = submaker.SubMaker() if self.srt_file else None
        if self.output_file is None:
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])

        else:
            with self.output_file.open("ab") as file:
                if self.srt_file is None:
                    async for chunk in com.stream():
                        if chunk["type"] == "audio":
                            file.write(chunk["data"])
                            chunks.append(chunk["data"])

                else:
                    async for chunk in com.stream():
                        if chunk["type"] == "audio":
                            file.write(chunk["data"])
                            chunks.append(chunk["data"])
                        elif chunk["type"] == "subtitle":
                            sub.feed(chunk)
                    self.srt_file.write_text(sub.get_srt(), encoding='utf-8')
        temp_chunks = b''.join(chunks)
        mp3 = AudioSegment.from_file(io.BytesIO(temp_chunks), format="mp3")
        pcm = np.frombuffer(mp3.raw_data, dtype=np.int16)
        print(pcm[:100], pcm.shape, pcm.dtype)
        #sd.play(pcm, samplerate=mp3.frame_rate)
        return chunks
    
    async def generate(self, text):
        '''
        Generate audio from text and play it.
        Args:
            text (str): Text to be converted to speech.
        '''
        reexp = r'(?<=[.!?。])\s*'
        chunks = re.split(reexp, text)
        p = pa.PyAudio()
        stream = p.open(format=pa.paInt16, channels=1, rate=24000, output=True)
        for chunk in chunks:
            if not chunk.strip():
                continue
            audio_chunks = await self.process_audio(chunk)
            mp3 = b''.join(audio_chunks)
            audio = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
            #stream.write(audio.raw_data)

if __name__ == "__main__":
    with open("tts/text.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
    for text in texts:
        text = text.strip()
        if not text:
            continue
        print(f"Processing: {text}")
        edge_tts = EdgeTTS(voice=VOICE, output_file=OUTPUT_FILE, srt_file=SRT_FILE)
        asyncio.run(edge_tts.generate(text))
        print(f"Finished processing: {text}")