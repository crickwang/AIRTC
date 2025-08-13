## Second Week Working Summary

[github](https://github.com/crickwang/Low-Latency-LLM)

---

### Implementation
- Implemented streaming ASR, LLM, and TTS models.
- Established WebRTC connection on the local machine.
- Designed simple front end page with HTML5, CSS, and JS. 

#### ASR
With streaming FunASR, the model can achieve the desired functionality where the transcription happens alongwith the user's voice inputs. The results are then passed to the next LLM model.

#### LLM
Simple API call of the Baidu LLM model.

#### TTS
With EdgeTTS, the transcription is fast and accurate. Transcription through chunks are implemented, the user may hear from the WebRTC Client while the transcription is happening in the background. 

### Challenges
- The main challenge is the poor performance of the FunASR, may possibly change to another more advanced ASR model.

### Next Steps
- Incorporate the existing ASR-LLM-TTS process with WebRTC, create an execuatble environment for voice communication. 
- Change the ASR model to more powerful online and paid model. 

