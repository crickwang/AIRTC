"""Unit tests for the energy-based VADs.

Pure numpy, no ML deps or network — safe to run in CI unlike tests/test_server.py.
Frames are TIME_PER_CHUNK = 10ms, so frame counts map directly to milliseconds.
"""

import numpy as np
import pytest

from vad.vad import MultiFrameVAD, SimpleVAD

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
