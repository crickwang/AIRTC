# AI-generated test suite (Claude) for the browser -> Google ASR -> LLM/browser leg of the
# pipeline, written after the 2026-09-06 prod cutover, when "Start Recording" showed up in
# the server log but nothing appeared on screen.
#
# GoogleASR.generate() is exercised with real av.AudioFrame objects run through the same
# AudioResampler and SimpleVAD the server uses, so the audio plumbing is real; only the
# network edge (google.cloud.speech_v1.SpeechClient) and the WebRTC edge (track and data
# channel) are faked. The assertions pin down the two outputs the rest of the system
# depends on: what is put on the LLM queue, and the exact JSON shape sent over the data
# channel -- webpage/static/js/main.js handleServerMessage() only renders a "transcription"
# message into the transcript box and only treats "stop_word" as a pause.

import asyncio
import datetime
import json
from types import SimpleNamespace

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler
from google.cloud import speech_v1 as speech

import clients.ASR.model as asr_model
import utils
from config.constants import ASR_SAMPLE_RATE, FORMAT, LAYOUT, VAD_THRESHOLD
from vad.vad import SimpleVAD

pytestmark = pytest.mark.asyncio

BROWSER_RATE = 48000  # what Chrome's Opus decode hands aiortc
FRAME_SAMPLES = 960  # 20 ms at 48 kHz, aiortc's frame size
LOUD = 20000  # SimpleVAD docs: noise < 2000, speech > 20000
STOP_WORD = "退出"  # the config.yaml default stop word
TRANSCRIPT = "can you hear me"


# ----------------------------------------------------------------------------- fakes


class FakeLogChannel:
    """Records data-channel sends as parsed JSON; readyState gates server_to_client()."""

    def __init__(self):
        self.sent = []
        self.readyState = "open"

    def send(self, message):
        self.sent.append(json.loads(message))

    def of_type(self, msg_type):
        return [m["message"] for m in self.sent if m["type"] == msg_type]


class FakeAudioPlayer:
    def __init__(self):
        self.interrupts = 0

    def request_interrupt(self):
        self.interrupts += 1


class FakeTrack:
    """A browser mic track: yields 20 ms frames, then ends like a hung-up call.

    recv() sleeps briefly per frame so the ASR worker thread gets scheduled between
    frames, the way real network pacing does.
    """

    def __init__(self, amplitudes):
        self._amplitudes = list(amplitudes)
        self._pts = 0

    async def recv(self):
        await asyncio.sleep(0.02)
        if not self._amplitudes:
            raise MediaStreamError()
        amplitude = self._amplitudes.pop(0)
        frame = make_frame(self._pts, amplitude)
        self._pts += FRAME_SAMPLES
        return frame


def make_frame(pts, amplitude):
    """One 48 kHz mono s16 frame: silence (0) or a 500 Hz square wave at `amplitude`.

    500 Hz survives the 48 kHz -> 24 kHz resample; a per-sample alternating wave would
    sit at 24 kHz and be filtered to silence, defeating the VAD check.
    """
    samples = np.zeros((1, FRAME_SAMPLES), dtype=np.int16)
    if amplitude:
        sign = np.where((np.arange(FRAME_SAMPLES) // 48) % 2 == 0, 1, -1)
        samples[0] = (sign * amplitude).astype(np.int16)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.sample_rate = BROWSER_RATE
    frame.pts = pts
    return frame


def final_response(text):
    """A single is_final streaming response, built from the real proto-plus types."""
    return speech.StreamingRecognizeResponse(
        results=[
            speech.StreamingRecognitionResult(
                alternatives=[speech.SpeechRecognitionAlternative(transcript=text, confidence=0.9)],
                is_final=True,
                result_end_time=datetime.timedelta(seconds=1, microseconds=650_000),
            )
        ]
    )


class FakeSpeechClient:
    """Stands in for google.cloud.speech_v1.SpeechClient.

    streaming_recognize() consumes exactly one audio request (proving PCM actually flowed
    from the track into the request stream) and then yields one final result, or raises
    to simulate a failed call to Google.
    """

    def __init__(self, transcript=TRANSCRIPT, error=None):
        self.transcript = transcript
        self.error = error
        self.calls = 0
        self.first_audio = []

    def streaming_recognize(self, streaming_config, requests):
        self.calls += 1
        if self.error is not None:
            raise self.error

        def responses():
            first = next(iter(requests))
            self.first_audio.append(first.audio_content)
            yield final_response(self.transcript)

        return responses()


# -------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_google(monkeypatch):
    """Patch the Google client constructor so GoogleASR() needs no credentials or network."""
    fake = FakeSpeechClient()
    monkeypatch.setattr(asr_model.speech, "SpeechClient", lambda: fake)
    return fake


@pytest.fixture
def pc():
    player = FakeAudioPlayer()
    return SimpleNamespace(log_channel=FakeLogChannel(), _audio_player=player)


def build_asr(pc):
    """Construct through the same factory path server.py uses (registry + kwargs)."""
    return utils.create_client(
        "asr",
        platform="google",
        rate=ASR_SAMPLE_RATE,
        language_code="zh-CN",
        alternative_language_codes=["en-US"],
        chunk_size=480,
        stop_word=STOP_WORD,
        pc=pc,
    )


async def run_pipeline(asr, track, pc, vad=None):
    """Drive GoogleASR.generate() to completion and return (llm_queue_items, interrupt_event)."""
    output_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    interrupt_event = asyncio.Event()
    resampler = AudioResampler(rate=ASR_SAMPLE_RATE, layout=LAYOUT, format=FORMAT)
    await asyncio.wait_for(
        asr.generate(
            track,
            output_queue,
            pc._audio_player,
            stop_event,
            interrupt_event,
            vad=vad,
            resampler=resampler,
            timeout=5,
        ),
        timeout=10,
    )
    items = []
    while not output_queue.empty():
        items.append(output_queue.get_nowait())
    return items, interrupt_event


# ----------------------------------------------------------------------------- tests


async def test_factory_builds_google_asr_when_credentials_resolve(fake_google, pc):
    # server.py calls create_client("asr", platform="google"); None here is exactly the
    # prod failure mode ("'NoneType' object has no attribute 'generate'").
    asr = build_asr(pc)
    assert isinstance(asr, asr_model.GoogleASR)
    assert asr.pc is pc


async def test_speech_is_transcribed_and_shown_in_browser(fake_google, pc):
    # One loud frame trips the VAD, silence follows while Google answers, then hang-up.
    asr = build_asr(pc)
    track = FakeTrack([LOUD] + [0] * 40)

    items, interrupt_event = await run_pipeline(
        asr, track, pc, vad=SimpleVAD(threshold=VAD_THRESHOLD)
    )

    # Audio really flowed, and in the format Google was configured for (LINEAR16 at
    # ASR_SAMPLE_RATE): the first request decodes as int16, is about one 20 ms frame long
    # at 24 kHz (the resampler's filter delay trims a little off the very first chunk),
    # and still carries the 500 Hz tone at roughly the amplitude the browser sent.
    assert fake_google.calls == 1
    pcm = np.frombuffer(fake_google.first_audio[0], dtype=np.int16)
    expected_samples = FRAME_SAMPLES * ASR_SAMPLE_RATE // BROWSER_RATE  # 960 -> 480
    assert 0.8 * expected_samples <= len(pcm) <= expected_samples
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    assert abs(rms - LOUD) / LOUD < 0.25

    # The LLM leg receives the transcript, then the end-of-stream sentinel.
    assert items == [TRANSCRIPT, None]

    # The browser leg receives it in the one shape main.js renders into the transcript box.
    transcription_msgs = [m for m in pc.log_channel.sent if m["type"] == "transcription"]
    assert transcription_msgs == [{"type": "transcription", "message": TRANSCRIPT}]
    assert "Listening to you..." in pc.log_channel.of_type("log")

    # Speech interrupts any playback in progress, and the interrupt is cleared afterwards.
    assert pc._audio_player.interrupts >= 1
    assert not interrupt_event.is_set()


async def test_silence_never_reaches_google_or_the_browser(fake_google, pc):
    # Quiet input below VAD_THRESHOLD: no session, no network call, nothing on screen.
    # This is what three of the four first prod calls looked like.
    asr = build_asr(pc)
    track = FakeTrack([0] * 20)

    items, _ = await run_pipeline(asr, track, pc, vad=SimpleVAD(threshold=VAD_THRESHOLD))

    assert fake_google.calls == 0
    assert items == [None]
    assert pc.log_channel.of_type("transcription") == []
    assert pc._audio_player.interrupts == 0


async def test_without_vad_every_frame_counts_as_speech(fake_google, pc):
    # vad=None is documented as "pass every frame through ASR": silence still transcribes.
    asr = build_asr(pc)
    track = FakeTrack([0] * 30)

    items, _ = await run_pipeline(asr, track, pc, vad=None)

    assert fake_google.calls >= 1
    assert items[0] == TRANSCRIPT
    assert TRANSCRIPT in pc.log_channel.of_type("transcription")


async def test_google_error_is_surfaced_to_browser_log_not_transcript(fake_google, pc):
    # A failing Google call must not silently produce nothing: the browser gets a "log"
    # line naming the error, and the LLM leg only ever sees the end sentinel.
    fake_google.error = RuntimeError("403 PERMISSION_DENIED: Cloud Speech-to-Text API")
    asr = build_asr(pc)
    track = FakeTrack([LOUD] + [0] * 20)

    items, _ = await run_pipeline(asr, track, pc, vad=SimpleVAD(threshold=VAD_THRESHOLD))

    assert fake_google.calls == 1
    assert items == [None]
    assert pc.log_channel.of_type("transcription") == []
    errors = [m for m in pc.log_channel.of_type("log") if "ASR thread error" in m]
    assert errors and "PERMISSION_DENIED" in errors[0]


async def test_stop_word_pauses_instead_of_transcribing(fake_google, pc):
    # main.js treats a "stop_word" message as "pause the session"; the word itself must
    # not be shown as a transcript or forwarded to the LLM.
    fake_google.transcript = STOP_WORD
    asr = build_asr(pc)
    track = FakeTrack([LOUD] + [0] * 40)

    items, _ = await run_pipeline(asr, track, pc, vad=SimpleVAD(threshold=VAD_THRESHOLD))

    assert pc.log_channel.of_type("stop_word") == ["Stop word triggered, pausing session"]
    assert pc.log_channel.of_type("transcription") == []
    assert STOP_WORD not in items
    assert pc._audio_player.interrupts >= 2  # once for speech onset, once for the stop word
