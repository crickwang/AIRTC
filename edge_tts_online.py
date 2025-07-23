import asyncio
import edge_tts.submaker as submaker
import edge_tts.communicate as communicate
import yaml

with open('tts_test.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
text = config.get('test_text', 'large_text').replace(' ', '').strip()
voice = config.get('edge_voice', 'en-GB-SoniaNeural')
output_file = config.get('edge_output', 'edge_tts_output.mp3')
srt_file = config.get('edge_subtitle', 'edge_tts_output.srt')

async def amain() -> None:
    com = communicate.Communicate(
        text=text, voice=voice)
    sub = submaker.SubMaker()
    with open(output_file, "wb") as file:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                sub.feed(chunk)

    with open(srt_file, "w", encoding="utf-8") as file:
        file.write(sub.get_srt())

if __name__ == "__main__":
    asyncio.run(amain())