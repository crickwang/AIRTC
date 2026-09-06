# VAD Assessment — 2026-08-04

Review of `vad\vad.py` and how it's wired into the ASR pipeline in `clients\ASR\model.py` and `server.py`.

## What's there now

Two energy-only detectors: `SimpleVAD` (the default, per `server.py:637`) compares a single frame's RMS energy against a fixed threshold, and `MultiFrameVAD` requires 3 consecutive loud frames. The threshold is a hardcoded absolute value (`VAD_THRESHOLD = 6000` in `config\constants.py:39`), and each ASR `generate()` loop calls `vad.is_speech(frame)` per WebRTC frame to decide when to start a transcription session and when to interrupt playback.

## What's inadequate

### Real bugs

1. **`MultiFrameVAD` never stops "speaking".** In `vad.py:59-82`, `is_currently_speaking` is set to `True` but there is no code path that ever resets it to `False`. After the first detection, `is_speech()` returns `True` for every frame forever — the hysteresis has an attack side but no release side. Also, `speech_frame_count` isn't zeroed when a quiet frame arrives, so 3 loud frames spread across minutes (three door slams) count as "consecutive" speech.

2. **The pre-speech frame buffer is dead code.** The `frames` list (`vad.py:46-57`) exists precisely to prevent the first frames from being swallowed, but nothing ever reads it — no ASR path flushes those buffered frames into the audio queue. So the problem the docstring warns about ("may swallow first several frames") is still unsolved, and even `SimpleVAD` clips the soft leading edge of an utterance because ASR only starts on the frame that already crossed the threshold.

3. **Silent failure in the factory.** `VADFactory.create` (`vad.py:123-128`) prints and returns `None` on error. Downstream, `if not vad or vad.is_speech(frame)` means a misconfigured VAD silently degrades to "every frame is speech" — each frame then fires `request_interrupt()` and spins up an ASR session.

### Design limitations

4. **Energy-only detection is inherently fragile.** Raw RMS against an absolute int16 amplitude can't distinguish speech from keyboard clicks, door slams, music, or the bot's own TTS echo, and the right threshold depends entirely on mic gain and distance. The docstrings even contradict each other about what the scale means (`MultiFrameVAD` says speech > 2000, `SimpleVAD` says speech > 20000, the constant is 6000).

5. **No adaptation.** There's no noise-floor estimation or calibration — a user in a café and a user in a quiet room get the same fixed 6000.

6. **End-of-speech logic lives outside the VAD, inconsistently.** The Whisper/Paraformer loops count `silence_count` against `max_silence_chunk`, the Google loop relies on API finality, and the VAD itself has no concept of a speech segment ending. Each ASR client re-implements turn-taking differently.

7. **Barge-in triggers on a single frame.** One loud frame immediately calls `audio_player.request_interrupt()` — a cough or speaker echo cuts off the assistant mid-sentence.

8. **Minor:** `MultiFrameVAD` doesn't call `super().__init__`, `vad.py` uses `print` instead of the project's logging setup, and there are no VAD tests in `tests\`.

## What would improve it

- **Adopt a model-based VAD.** The two standard options are **Silero VAD** (small ONNX/torch model, very accurate, ~1ms per chunk) and **py-webrtcvad** (GMM-based, extremely lightweight). Notably, the project already depends on **FunASR**, which ships **FSMN-VAD** — it could be registered as a third algorithm with zero new dependencies. Any of these slots cleanly into the existing `register`/factory pattern.
- **Fix the state machine.** Give the VAD proper onset (N consecutive speech frames) *and* release (M frames of hangover silence) transitions, and have it expose speech-start/speech-end events so the ASR loops stop rolling their own `silence_count` logic.
- **Actually use the pre-roll buffer.** Keep a small ring buffer of recent frames and flush it into the ASR audio queue at speech onset, so the first syllable isn't clipped.
- **Adaptive thresholding.** Track a rolling noise-floor estimate (e.g., exponential moving average of quiet-frame energy) and detect speech as a relative SNR jump instead of an absolute 6000, or at minimum calibrate for a second at session start.
- **Harden barge-in.** Require sustained speech (a few consecutive frames) before interrupting playback, and consider raising the threshold while TTS is playing to resist echo — browser-side `echoCancellation` on `getUserMedia` helps but isn't sufficient on speakers.
- **Cheap accuracy wins if staying energy-based:** add zero-crossing rate and band-limited energy (300–3000 Hz) to reject clicks and hum.
- **Housekeeping:** make the factory raise instead of returning `None`, expose threshold/frames via CLI args or settings, switch to `logging`, and add unit tests with synthetic frames (silence, white noise, tone bursts) — these are pure-numpy and could run in CI unlike `test_server.py`.

## Suggested starting point

Fix the `MultiFrameVAD` release bug and wire the pre-roll buffer first (those change behavior users can hear), then add a Silero or FSMN-VAD backend behind the existing `--vad` flag.
