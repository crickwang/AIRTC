# AI-generated test suite (Claude) for the leg that carries audio back to the browser:
# LLM text -> AzureTTS -> audio_queue -> AudioPlayer (the aiortc track in the SDP answer)
# -> the frames aiortc Opus-encodes and sends over WebRTC.
#
# Only the Azure Speech SDK is faked (a synthesizer that fires the same synthesizing /
# synthesis_completed / synthesis_canceled events the real one does, from a worker
# thread). Everything after that is the production code: AzureTTS.generate() with the
# server's SAMPLES_PER_FRAME, the real AudioPlayer built with the server's constructor
# arguments, and a consumer that calls player.recv() exactly the way aiortc does. The
# assertions pin down what the browser depends on: 10 ms mono s16 frames at 24 kHz with
# monotonic timestamps, the synthesized samples arriving intact and in order, a track
# that stays alive (silence, not an error) once speech ends, and playback that stops
# immediately on interrupt.

import asyncio
import json
import time
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import clients.TTS.model as tts_model
import utils
from audio_player.audio_player import AudioPlayer
from config.constants import AUDIO_SAMPLE_RATE, FORMAT, LAYOUT, SAMPLES_PER_FRAME

pytestmark = pytest.mark.asyncio

TONE_HZ = 440
TONE_MS = 300
TONE_AMPLITUDE = 12000
SDK_CHUNK_BYTES = 4096  # Azure streams PCM in a few KB per synthesizing event
PLAYER_QUEUE_MAXSIZE = 100  # server.py: asyncio.Queue(maxsize=100)


# ----------------------------------------------------------------------------- fakes


class FakeLogChannel:
    def __init__(self):
        self.sent = []
        self.readyState = "open"

    def send(self, message):
        self.sent.append(json.loads(message))

    def logs(self):
        return [m["message"] for m in self.sent if m["type"] == "log"]


class FakeSignal:
    """Mimics the SDK's EventSignal: connect(handler), disconnect_all(), and firing."""

    def __init__(self):
        self.handlers = []

    def connect(self, handler):
        self.handlers.append(handler)

    def disconnect_all(self):
        self.handlers.clear()

    def fire(self, evt):
        for handler in list(self.handlers):
            handler(evt)


def tone_pcm(text):
    """Deterministic PCM for a sentence: a 440 Hz sine, TONE_MS long, 24 kHz s16 mono."""
    n = AUDIO_SAMPLE_RATE * TONE_MS // 1000
    t = np.arange(n) / AUDIO_SAMPLE_RATE
    return (np.sin(2 * np.pi * TONE_HZ * t) * TONE_AMPLITUDE).astype(np.int16)


class FakeSynthesizer:
    """Stands in for speechsdk.SpeechSynthesizer.

    speak_text_async() streams tone_pcm(text) through the synthesizing signal in
    SDK-sized chunks, then fires synthesis_completed -- or synthesis_canceled with an
    error, to simulate Azure rejecting the call (bad key, wrong region, no network).
    """

    def __init__(self):
        self.synthesizing = FakeSignal()
        self.synthesis_completed = FakeSignal()
        self.synthesis_canceled = FakeSignal()
        self.spoken = []
        self.fail_with = None

    def speak_text_async(self, text):
        self.spoken.append(text)
        if self.fail_with is not None:
            details = SimpleNamespace(error_details=self.fail_with)
            self.synthesis_canceled.fire(
                SimpleNamespace(result=SimpleNamespace(cancellation_details=details))
            )
            return
        pcm = tone_pcm(text).tobytes()
        for i in range(0, len(pcm), SDK_CHUNK_BYTES):
            evt = SimpleNamespace(result=SimpleNamespace(audio_data=pcm[i : i + SDK_CHUNK_BYTES]))
            self.synthesizing.fire(evt)
        self.synthesis_completed.fire(SimpleNamespace(result=SimpleNamespace(reason="completed")))


class FakeSpeechConfig:
    def __init__(self, subscription, region):
        self.subscription = subscription
        self.region = region
        self.speech_synthesis_voice_name = None
        self.output_format = None

    def set_speech_synthesis_output_format(self, fmt):
        self.output_format = fmt


# -------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake_azure(monkeypatch):
    """Replace the Azure SDK module inside clients.TTS.model with an in-process fake."""
    synth = FakeSynthesizer()
    sdk = SimpleNamespace(
        SpeechConfig=FakeSpeechConfig,
        SpeechSynthesisOutputFormat=SimpleNamespace(Raw24Khz16BitMonoPcm="raw-24k-16bit-mono"),
        SpeechSynthesizer=lambda speech_config, audio_config: synth,
        Connection=SimpleNamespace(
            from_speech_synthesizer=lambda s: SimpleNamespace(open=lambda pre: None)
        ),
        ResultReason=SimpleNamespace(SynthesizingAudioCompleted="completed"),
    )
    monkeypatch.setattr(tts_model, "speechsdk", sdk)
    return synth


@pytest.fixture
def pc():
    return SimpleNamespace(log_channel=FakeLogChannel())


def build_tts(pc):
    """Construct through the same factory path server.py uses (registry + kwargs)."""
    return utils.create_client(
        "tts",
        platform="azure",
        voice="zh-CN-YunyangNeural",
        key="not-a-real-key",
        region="eastus",
        pc=pc,
    )


def build_player():
    """The AudioPlayer exactly as server.py attaches it to the peer connection."""
    audio_queue = asyncio.Queue(maxsize=PLAYER_QUEUE_MAXSIZE)
    player = AudioPlayer(
        audio_queue,
        sample_rate=AUDIO_SAMPLE_RATE,
        samples_per_frame=SAMPLES_PER_FRAME,
        format=FORMAT,
        layout=LAYOUT,
    )
    return audio_queue, player


def run_tts(tts, llm_queue, audio_queue, stop_event, interrupt_event):
    return asyncio.create_task(
        tts.generate(
            llm_queue,
            audio_queue,
            stop_event,
            interrupt_event,
            samples_per_frame=SAMPLES_PER_FRAME,
            timeout=2,
        )
    )


async def pull_frames(player, count):
    """Consume the track the way aiortc's sender does: recv() in a loop."""
    return [await asyncio.wait_for(player.recv(), timeout=5) for _ in range(count)]


def is_silent(frame):
    return not np.any(frame.to_ndarray())


def samples_of(frames):
    return np.concatenate([f.to_ndarray()[0] for f in frames]) if frames else np.array([], np.int16)


# ----------------------------------------------------------------------------- tests


async def test_factory_builds_azure_tts_with_24khz_pcm_output(fake_azure, pc):
    # server.py calls create_client("tts", platform="azure"); the output format it asks
    # Azure for must match AUDIO_SAMPLE_RATE, or the browser hears the wrong pitch.
    tts = build_tts(pc)
    assert isinstance(tts, tts_model.AzureTTS)
    assert tts.config.output_format == "raw-24k-16bit-mono"
    assert AUDIO_SAMPLE_RATE == 24000
    assert tts.config.speech_synthesis_voice_name == "zh-CN-YunyangNeural"


async def test_tts_audio_reaches_browser_as_10ms_frames_intact(fake_azure, pc):
    # One LLM sentence -> Azure PCM -> audio_queue -> AudioPlayer.recv() frames, which
    # is what aiortc encodes and sends. The frames must be exactly what the browser's
    # Opus decoder expects, and the samples must be the synthesized ones, in order.
    tts = build_tts(pc)
    audio_queue, player = build_player()
    llm_queue = asyncio.Queue()
    stop_event, interrupt_event = asyncio.Event(), asyncio.Event()
    tts_task = run_tts(tts, llm_queue, audio_queue, stop_event, interrupt_event)

    await llm_queue.put("As a large language model")
    expected = tone_pcm("As a large language model")
    expected_frames = len(expected) // SAMPLES_PER_FRAME
    assert len(expected) % SAMPLES_PER_FRAME == 0, "test tone should be a whole number of frames"

    frames = await pull_frames(player, expected_frames)
    await llm_queue.put(None)  # LLM finished
    await asyncio.wait_for(tts_task, timeout=5)

    # Frame format the browser depends on: 10 ms, mono, s16, 24 kHz, monotonic pts.
    for f in frames:
        assert f.sample_rate == AUDIO_SAMPLE_RATE
        assert f.samples == SAMPLES_PER_FRAME == AUDIO_SAMPLE_RATE // 100
        assert f.format.name == FORMAT and f.layout.name == LAYOUT
        assert f.time_base == Fraction(1, AUDIO_SAMPLE_RATE)
    assert [f.pts for f in frames] == [i * SAMPLES_PER_FRAME for i in range(expected_frames)]

    # The audio itself: every synthesized sample, bit-exact and in order.
    assert all(not is_silent(f) for f in frames)
    np.testing.assert_array_equal(samples_of(frames), expected)

    # Azure was asked to speak exactly that sentence, and the browser's log panel was told.
    assert fake_azure.spoken == ["As a large language model"]
    assert any("Starting streaming for sentence 1" in m for m in pc.log_channel.logs())
    assert "TTS: end of processing" in pc.log_channel.logs()


async def test_track_stays_alive_with_silence_after_speech_ends(fake_azure, pc):
    # After the LLM's end marker, TTS posts None on the audio queue. The player must
    # keep answering recv() with silent frames (continuing pts), never raise or stall
    # for good: a WebRTC sender that gets an exception tears the track down.
    tts = build_tts(pc)
    audio_queue, player = build_player()
    llm_queue = asyncio.Queue()
    stop_event, interrupt_event = asyncio.Event(), asyncio.Event()
    tts_task = run_tts(tts, llm_queue, audio_queue, stop_event, interrupt_event)

    await llm_queue.put("Short answer.")
    n = len(tone_pcm("Short answer.")) // SAMPLES_PER_FRAME
    frames = await pull_frames(player, n)
    await llm_queue.put(None)
    await asyncio.wait_for(tts_task, timeout=5)

    # The queue now holds only the end marker; the next frames are silence, still paced
    # and still numbered, so the browser hears a quiet line rather than a dropped call.
    after = await pull_frames(player, 2)
    assert all(is_silent(f) for f in after)
    assert after[0].pts == frames[-1].pts + SAMPLES_PER_FRAME
    assert after[1].pts == after[0].pts + SAMPLES_PER_FRAME
    assert audio_queue.empty()


async def test_interrupt_drops_buffered_speech_immediately(fake_azure, pc):
    # The user talking over the assistant triggers request_interrupt() (from the ASR
    # leg or the browser's clear_audio message). Whatever is buffered must not be
    # played: the very next frame is silence and the queue is drained.
    audio_queue, player = build_player()
    pcm = tone_pcm("a long sentence the user interrupts")
    for i in range(0, len(pcm), SAMPLES_PER_FRAME):
        await audio_queue.put(pcm[i : i + SAMPLES_PER_FRAME])
    queued = audio_queue.qsize()

    first = await pull_frames(player, 3)
    assert all(not is_silent(f) for f in first)
    player.request_interrupt()

    nxt = (await pull_frames(player, 1))[0]
    assert is_silent(nxt)
    assert audio_queue.qsize() == 0, f"{queued} chunks were queued; interrupt must drop them"
    assert nxt.pts == first[-1].pts + SAMPLES_PER_FRAME


async def test_sentences_stream_back_in_order(fake_azure, pc):
    # The LLM hands TTS one sentence at a time; the browser must hear them in order
    # with nothing lost at the boundary.
    tts = build_tts(pc)
    audio_queue, player = build_player()
    llm_queue = asyncio.Queue()
    stop_event, interrupt_event = asyncio.Event(), asyncio.Event()
    tts_task = run_tts(tts, llm_queue, audio_queue, stop_event, interrupt_event)

    await llm_queue.put("First sentence.")
    await llm_queue.put("Second sentence.")
    expected = np.concatenate([tone_pcm("First sentence."), tone_pcm("Second sentence.")])
    frames = await pull_frames(player, len(expected) // SAMPLES_PER_FRAME)
    await llm_queue.put(None)
    await asyncio.wait_for(tts_task, timeout=5)

    assert fake_azure.spoken == ["First sentence.", "Second sentence."]
    np.testing.assert_array_equal(samples_of(frames), expected)


async def test_frames_are_paced_in_real_time(fake_azure, pc):
    # aiortc sends a frame as soon as recv() returns, so the player must release one
    # 10 ms frame per 10 ms or the browser's jitter buffer overflows and drops audio.
    audio_queue, player = build_player()
    pcm = tone_pcm("pacing")
    for i in range(0, len(pcm), SAMPLES_PER_FRAME):
        await audio_queue.put(pcm[i : i + SAMPLES_PER_FRAME])

    t0 = time.monotonic()
    frames = await pull_frames(player, 20)
    elapsed = time.monotonic() - t0

    assert all(not is_silent(f) for f in frames)
    assert elapsed >= 19 * SAMPLES_PER_FRAME / AUDIO_SAMPLE_RATE * 0.8  # ~190 ms, with slack


async def test_azure_failure_yields_no_audio_but_pipeline_still_finishes(fake_azure, pc):
    # If Azure rejects the request (bad key / region), the browser must not be left
    # hanging: TTS finishes, posts its end marker, and no non-silent frame is emitted.
    fake_azure.fail_with = "WebSocket upgrade failed: Authentication error (401)"
    tts = build_tts(pc)
    audio_queue, player = build_player()
    llm_queue = asyncio.Queue()
    stop_event, interrupt_event = asyncio.Event(), asyncio.Event()
    tts_task = run_tts(tts, llm_queue, audio_queue, stop_event, interrupt_event)

    await llm_queue.put("This will fail.")
    await llm_queue.put(None)
    await asyncio.wait_for(tts_task, timeout=5)

    assert fake_azure.spoken == ["This will fail."]
    assert audio_queue.qsize() == 1 and audio_queue.get_nowait() is None  # end marker only
    frame = (await pull_frames(player, 1))[0]
    assert is_silent(frame)
    assert "TTS: end of processing" in pc.log_channel.logs()
