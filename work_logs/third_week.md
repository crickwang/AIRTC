## Third Week Working Summary

[github](https://github.com/crickwang/Low-Latency-LLM)

---

### Implementation
- Incorporated ASR-LLM-TTS process with WebRTC. Created partially executable workflow.
- Switched ASR model to Google STT service. 
- Implemented more features in pausing and resuming recording from WebRTC client. 

#### ASR
Switched to Google STT, require additional setups about environment and tokens.

### Challenges
Encountered a bug where audio is playable on the Server side, recorded on the Client side, but failed to make sound on the Client side. 
- MediaTrack influenced the audio playback. :x:
- Different sample rate confused peer connection in WebRTC. :x:
- Audio was empty / Audio was disconfigured by parameters, including sample rate, timestamp, counter, etc. :x:
    > Note: The audio is rockin' good right before returning in recv() function and feed in WebRTC Client.
- Created a dummy MediaTrack that do not recieve track from the queue generated from TTS. Instead, the function itself generated the sine waveform and play it on the Client side. :white_check_mark: Not the problem of the Client. :x:
- Timeframe was too small that each audio chunk raced with each other. :x: Created a buffer that merges all the chunks. Still no sound.
- Client recieved nothing. :x: It did.

### Next Steps
- **URGENT: solve this bug.**
- Transform the EdgeTTS into API call that does streaming output alongwith LLM output. Possibly Google TTS or Microsoft Azure. 