"""Unit tests for the energy-based VADs.

Pure numpy, no ML deps or network — safe to run in CI unlike tests/test_server.py.
Frames are TIME_PER_CHUNK = 10ms, so frame counts map directly to milliseconds.
"""

import numpy as np
import pytest

from vad.vad import MultiFrameVAD, SimpleVAD, VADFactory

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

    def test_simple_vad_has_no_preroll(self):
        assert SimpleVAD(THRESHOLD).drain_preroll() == []


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
