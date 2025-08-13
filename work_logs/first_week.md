## First Week Working Summary

[github](https://github.com/crickwang/Low-Latency-LLM)

---

### Exploration
- Explored the state-of-the-art open source models for Automatic Speech Recognition (ASR), Large Language Models (LLMs), and Text-to-Speech (TTS). Hoping to combine these technologies for a speech2speech (s2s) solution.'
- Models:
  - ASR: Whisper
  - LLM: DeepSeek, Llama 3 (local), ernie-4.5-0.3b (online)
  - TTS: Kokoro, Edge_tts
- Implemented a basic pipeline to convert speech to text, process the text with an LLM, and convert the text back to speech. Without the streaming feature, the pipeline works but is not efficient. Whisper takes about 30 seconds to transcribe a 30-second audio clip, and the LLM takes about 3 seconds to process the text. The TTS model takes about 3 seconds to generate speech from text. Although with good quality, the overall latency is around 36 seconds, far from the requirement of real-time processing.

### Challenges
- The main challenge is the latency of the current pipeline, which is not suitable for real-time applications.
- For TTS, Kokoro turns out have poor audio quality, while Edge_tts has better quality. Although the AI voice of Edge_tts makes it kind of robotic, it is still acceptable for the current stage compared with Kokoro.
- For ASR, Whisper is not optimized for real-time processing, and it takes too long to transcribe audio. The latency is still unacceptable for real-time applications even with parallel processing.

### Next Steps
- Investigate ways to reduce the latency of the ASR model, possibly by exploring other models like FunASR.
- Explore more efficient LLMs that can process text faster, possibly open source.
- Investigate alternative TTS models that can provide better audio quality, less robotic speech and lower latency, possibly by parallel processing or using smaller models.

