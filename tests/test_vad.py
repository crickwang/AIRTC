"""Unit tests for the VAD implementations.

Pure numpy, no network and no model weights — safe to run in CI unlike
tests/test_server.py. The FSMN tests drive the streaming/chunking logic against a
scripted stand-in for funasr's AutoModel, so they cover the integration but not the
model's own accuracy; only the resampling test needs torch and skips without it.
Frames are TIME_PER_CHUNK = 10ms, so frame counts map directly to milliseconds.
"""

import numpy as np
import pytest

from vad.vad import FSMNVAD, MultiFrameVAD, SimpleVAD, VADFactory

THRESHOLD = 6000


def loud(samples=240, amplitude=20000):
    """A frame whose RMS is comfortably above THRESHOLD."""
    return np.full(samples, amplitude, dtype=np.int16)


def quiet(samples=240, amplitude=100):
    """A frame whose RMS is comfortably below THRESHOLD."""
    return np.full(samples, amplitude, dtype=np.int16)


def feed(vad, frame, count):
    """Push the same frame in count times, returning the final verdict."""
    result = False
    for _ in range(count):
        result = vad.is_speech(frame)
    return result


class TestSimpleVAD:
    def test_loud_frame_is_speech(self):
        assert SimpleVAD(THRESHOLD).is_speech(loud())

    def test_quiet_frame_is_not_speech(self):
        assert not SimpleVAD(THRESHOLD).is_speech(quiet())

    def test_empty_frame_is_not_speech(self):
        assert not SimpleVAD(THRESHOLD).is_speech(np.array([], dtype=np.int16))


class TestMultiFrameVADOnset:
    def test_requires_consecutive_loud_frames(self):
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3)
        assert not vad.is_speech(loud())
        assert not vad.is_speech(loud())
        assert vad.is_speech(loud())

    def test_non_consecutive_loud_frames_do_not_trigger(self):
        """Three door slams separated by silence are not speech."""
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3)
        for _ in range(3):
            assert not vad.is_speech(loud())
            assert not vad.is_speech(quiet())
        assert not vad.is_currently_speaking

    def test_empty_frame_is_not_speech(self):
        vad = MultiFrameVAD(THRESHOLD)
        assert not vad.is_speech(np.array([], dtype=np.int16))


class TestMultiFrameVADRelease:
    """Regression tests for the bug where is_currently_speaking was never reset."""

    def test_releases_after_silence_hangover(self):
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, silence_frames_required=10)
        assert feed(vad, loud(), 3)

        # Still speaking while inside the hangover window.
        assert feed(vad, quiet(), 9)
        # The 10th consecutive quiet frame closes the segment.
        assert not vad.is_speech(quiet())

    def test_stays_silent_after_release(self):
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, silence_frames_required=10)
        feed(vad, loud(), 3)
        feed(vad, quiet(), 10)
        assert not feed(vad, quiet(), 50)

    def test_brief_dip_does_not_end_speech(self):
        """Energy dips between syllables must not close the segment."""
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, silence_frames_required=10)
        feed(vad, loud(), 3)
        for _ in range(20):
            assert feed(vad, quiet(), 5)   # 50ms dip, under the 100ms hangover
            assert vad.is_speech(loud())   # loud frame cancels the release

    def test_can_retrigger_after_release(self):
        """A second utterance is detected after the first one ends."""
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, silence_frames_required=10)
        feed(vad, loud(), 3)
        feed(vad, quiet(), 10)
        assert not vad.is_currently_speaking
        assert feed(vad, loud(), 3)

    def test_turn_can_end_in_asr_loop(self):
        """The condition clients/ASR/model.py relies on to close a turn.

        It counts consecutive frames where is_speech() is False and ends the turn
        once that exceeds max_silence_chunk (50 frames / 500ms). Before the release
        fix this loop never terminated.
        """
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, silence_frames_required=10)
        feed(vad, loud(), 3)

        max_silence_chunk = 50
        silence_count = 0
        for _ in range(500):
            if not vad.is_speech(quiet()):
                silence_count += 1
            else:
                silence_count = 0
            if silence_count > max_silence_chunk:
                break
        else:
            pytest.fail("turn never ended: is_speech() stayed True forever")

        # 10 frames of hangover, then 51 to exceed max_silence_chunk.
        assert silence_count == max_silence_chunk + 1


class TestPreroll:
    """The frames buffer used to be populated but never read, so the audio before
    the detection point was dropped and the first syllable got clipped."""

    def test_preroll_returns_frames_before_detection(self):
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, preroll_frames=10)
        # Enough quiet lead-in to saturate the buffer, then the 3 loud frames
        # that confirm speech.
        feed(vad, quiet(), 20)
        assert feed(vad, loud(), 3)

        preroll = vad.drain_preroll()
        # Buffer holds 10 frames; the newest is the one just evaluated, excluded here.
        assert len(preroll) == 9
        # Oldest first: 7 quiet frames of lead-in, then the first 2 loud frames.
        assert all(np.array_equal(f, quiet()) for f in preroll[:7])
        assert all(np.array_equal(f, loud()) for f in preroll[7:])

    def test_preroll_excludes_current_frame(self):
        """The caller queues the triggering frame itself, so it must not be duplicated."""
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, preroll_frames=10)
        feed(vad, quiet(), 20)
        feed(vad, loud(), 3)
        assert len(vad.drain_preroll()) == len(vad.frames) - 1

    def test_draining_clears_the_buffer(self):
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=3, preroll_frames=10)
        feed(vad, quiet(), 6)
        feed(vad, loud(), 3)
        assert vad.drain_preroll()
        assert vad.drain_preroll() == []

    def test_buffer_always_covers_the_onset_window(self):
        """A small preroll_frames must not drop the frames that triggered detection."""
        vad = MultiFrameVAD(THRESHOLD, speech_frames_required=5, preroll_frames=2)
        assert len(vad.frames) == 5
        assert feed(vad, loud(), 5)
        assert len(vad.drain_preroll()) == 4

    def test_simple_vad_buffers_preroll(self):
        """SimpleVAD fires on the frame that crosses the threshold, so without a
        look-back the quieter onset before it would be dropped."""
        vad = SimpleVAD(THRESHOLD, preroll_frames=10)
        feed(vad, quiet(), 20)
        assert vad.is_speech(loud())
        preroll = vad.drain_preroll()
        assert len(preroll) == 9
        assert all(np.array_equal(f, quiet()) for f in preroll)

    def test_simple_vad_preroll_can_be_disabled(self):
        assert SimpleVAD(THRESHOLD, preroll_frames=0).drain_preroll() == []

    def test_simple_vad_returns_plain_bool(self):
        """The ASR loops branch on this; numpy scalars would still work but the
        other VADs return bool, so keep it consistent."""
        assert SimpleVAD(THRESHOLD).is_speech(loud()) is True
        assert SimpleVAD(THRESHOLD).is_speech(quiet()) is False


class TestVADFactory:
    """create() used to swallow errors and return None. Downstream that reads as
    'every frame is speech', firing an interrupt and an ASR session per frame."""

    def test_creates_registered_vad(self):
        assert isinstance(VADFactory.create("simple", threshold=THRESHOLD), SimpleVAD)
        assert isinstance(VADFactory.create("multiFrame", threshold=THRESHOLD), MultiFrameVAD)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown VAD algorithm"):
            VADFactory.create("definitely-not-a-vad", threshold=THRESHOLD)

    def test_error_lists_available_algorithms(self):
        with pytest.raises(ValueError, match="simple"):
            VADFactory.create("typo", threshold=THRESHOLD)

    def test_bad_kwargs_raise(self):
        with pytest.raises(TypeError):
            VADFactory.create("simple", not_a_real_parameter=1)


class FakeFSMNModel:
    """Stands in for funasr's AutoModel so the integration logic can be tested
    without downloading weights. Returns the scripted segment list per call."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        value = self.script.pop(0) if self.script else []
        return [{"key": "fake", "value": value}]


@pytest.fixture
def fsmn(monkeypatch):
    """Build an FSMNVAD wired to a scripted fake model, at the model's own rate so
    no resampling (and therefore no torch) is involved."""

    def _build(script=None, **kwargs):
        model = FakeFSMNModel(script)
        monkeypatch.setattr(FSMNVAD, "_load_model", staticmethod(lambda *a, **k: model))
        monkeypatch.setattr(FSMNVAD, "_build_resampler", staticmethod(lambda *a, **k: None))
        kwargs.setdefault("input_sample_rate", FSMNVAD.MODEL_SAMPLE_RATE)
        return FSMNVAD(**kwargs), model

    return _build


class TestFSMNVADChunking:
    def test_model_not_run_until_a_full_chunk_arrives(self, fsmn):
        # 60ms chunks at 16kHz = 960 samples = 6 frames of 10ms.
        vad, model = fsmn(chunk_size_ms=60)
        frame = quiet(samples=160)
        for _ in range(5):
            vad.is_speech(frame)
        assert model.calls == []
        vad.is_speech(frame)
        assert len(model.calls) == 1

    def test_chunk_handed_to_model_is_the_configured_length(self, fsmn):
        vad, model = fsmn(chunk_size_ms=60)
        feed(vad, quiet(samples=160), 6)
        assert len(model.calls[0]["input"]) == vad.chunk_samples

    def test_audio_is_normalised_to_unit_range(self, fsmn):
        """The funasr frontend scales by 1<<15, so it expects [-1, 1] not int16."""
        vad, model = fsmn(chunk_size_ms=60)
        feed(vad, loud(samples=160), 6)
        audio = model.calls[0]["input"]
        assert audio.dtype == np.float32
        assert np.allclose(audio, 20000 / 32768.0)
        assert np.abs(audio).max() <= 1.0

    def test_model_told_the_rate_it_expects(self, fsmn):
        vad, model = fsmn(chunk_size_ms=60)
        feed(vad, quiet(samples=160), 6)
        assert model.calls[0]["fs"] == FSMNVAD.MODEL_SAMPLE_RATE
        assert model.calls[0]["is_final"] is False

    def test_cache_is_reused_across_chunks(self, fsmn):
        """Streaming state lives in the cache; a fresh one each call would reset it."""
        vad, model = fsmn(chunk_size_ms=60)
        feed(vad, quiet(samples=160), 12)
        assert len(model.calls) == 2
        assert model.calls[0]["cache"] is model.calls[1]["cache"]


class TestFSMNVADState:
    def test_open_segment_starts_speech(self, fsmn):
        vad, _ = fsmn(script=[[[100, -1]]], chunk_size_ms=60)
        assert feed(vad, quiet(samples=160), 6)

    def test_close_segment_ends_speech(self, fsmn):
        vad, _ = fsmn(script=[[[100, -1]], [], [[-1, 500]]], chunk_size_ms=60)
        frame = quiet(samples=160)
        assert feed(vad, frame, 6)     # [beg, -1] opens
        assert feed(vad, frame, 6)     # [] leaves it open
        assert not feed(vad, frame, 6)  # [-1, end] closes

    def test_self_contained_segment_ends_speech(self, fsmn):
        vad, _ = fsmn(script=[[[100, 400]]], chunk_size_ms=60)
        assert not feed(vad, quiet(samples=160), 6)

    def test_state_holds_between_inferences(self, fsmn):
        """Frames arrive faster than the model runs, so the last verdict must persist."""
        vad, _ = fsmn(script=[[[100, -1]]], chunk_size_ms=60)
        frame = quiet(samples=160)
        assert feed(vad, frame, 6)
        for _ in range(5):
            assert vad.is_speech(frame)

    def test_empty_frame_reports_state_without_buffering(self, fsmn):
        vad, model = fsmn(script=[[[100, -1]]], chunk_size_ms=60)
        feed(vad, quiet(samples=160), 6)
        pending = vad.pending_samples
        assert vad.is_speech(np.array([], dtype=np.int16))
        assert vad.pending_samples == pending
        assert len(model.calls) == 1

    def test_preroll_is_inherited(self, fsmn):
        vad, _ = fsmn(chunk_size_ms=60, preroll_frames=20)
        feed(vad, quiet(samples=160), 30)
        assert len(vad.drain_preroll()) == 19


class TestFSMNVADResampling:
    def test_no_resampler_at_native_rate(self):
        assert FSMNVAD._build_resampler(FSMNVAD.MODEL_SAMPLE_RATE) is None

    def test_resamples_when_pipeline_rate_differs(self, monkeypatch):
        """The pipeline feeds 24kHz (ASR_SAMPLE_RATE) but the model is 16kHz-only."""
        torch = pytest.importorskip("torch")
        model = FakeFSMNModel()
        calls = []

        def fake_resample(tensor):
            calls.append(len(tensor))
            return torch.from_numpy(np.zeros(960, dtype=np.float32))

        monkeypatch.setattr(FSMNVAD, "_load_model", staticmethod(lambda *a, **k: model))
        monkeypatch.setattr(FSMNVAD, "_build_resampler", staticmethod(lambda *a, **k: fake_resample))
        vad = FSMNVAD(input_sample_rate=24000, chunk_size_ms=60)

        # 60ms at 24kHz = 1440 samples = 6 frames of 240.
        assert vad.chunk_samples == 1440
        feed(vad, quiet(samples=240), 6)
        assert calls == [1440]
        assert len(model.calls[0]["input"]) == 960  # 60ms at 16kHz
