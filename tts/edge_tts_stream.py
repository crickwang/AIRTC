import asyncio
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
import pyaudio as pa
from pathlib import Path
from pydub import AudioSegment
import re
import io

VOICE = "zh-CN-XiaoxiaoNeural"
OUTPUT_FILE = "audio/output.mp3"
SRT_FILE = "audio/output.srt"

class EdgeTTS:
    def __init__(self, voice="zh-CN-XiaoxiaoNeural", output_file=None, srt_file=None):
        self.voice = voice
        if output_file is not None:
            self.output_file = Path(output_file)
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.output_file = None
        if srt_file is not None:
            self.srt_file = Path(srt_file)
            self.srt_file.parent.mkdir(parents=True, exist_ok=True)
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
            with self.output_file.open("wb") as file:
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
            stream.write(audio.raw_data)

if __name__ == "__main__":
    edge_tts = EdgeTTS(voice="zh-CN-XiaoxiaoNeural")
    asyncio.run(edge_tts.generate("你好，你在干嘛，我觉得意大利面需要混72号混凝土。"))