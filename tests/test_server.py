# AI-generated test suite (Claude) for the warm-up / pre-connect flow in server.py,
# written on the `testing` branch.
#
# These tests exercise WebPage._activate_session, WebPage._close_if_never_activated,
# and WebPage._drain_track directly, with a fake peer-connection object and a fake
# create_client() so no real aiortc negotiation, network I/O, or ASR/LLM/TTS client
# construction happens. The quota/guest logic runs against a real (tmp) sqlite DB via
# auth_store, mirroring tests/test_auth_store.py's isolation fixture.

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

import auth_store
import server

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point auth_store at a throwaway sqlite file so tests never touch a real auth.db."""
    monkeypatch.setattr(auth_store, "DB_PATH", tmp_path / "test_auth.db")
    monkeypatch.setattr(auth_store, "APP_ENV", "development")
    auth_store.init_auth_db()


@pytest.fixture(autouse=True)
def fake_pipeline_clients(monkeypatch):
    """Replace ASR/LLM/TTS/VAD construction with an instant no-op client."""
    monkeypatch.setattr(server, "create_client", fake_create_client)


class FakeClient:
    """Stands in for a real ASR/LLM/TTS/VAD client — completes instantly, no network."""

    async def generate(self, *args, **kwargs):
        return


def fake_create_client(model_type, platform=None, **kwargs):
    return FakeClient()


class FakeLogChannel:
    """Records everything sent over the data channel so tests can assert on it.

    readyState mimics aiortc's RTCDataChannel — utils.server_to_client() checks it
    before sending the "Start Recording" log line on successful activation.
    """

    def __init__(self):
        self.sent = []
        self.readyState = "open"

    def send(self, message):
        self.sent.append(json.loads(message))


class FakeTrack:
    """A client audio track whose recv() hangs until the test lets it go."""

    def __init__(self):
        self._release = asyncio.Event()

    async def recv(self):
        await self._release.wait()
        raise server.MediaStreamError


class FakePC:
    """A stand-in peer connection — a plain object (hashable, unlike SimpleNamespace)
    since WebPage.pcs is a set."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_pc(**overrides):
    defaults = dict(
        _activated=False,
        _client_track=FakeTrack(),
        _user=None,
        _guest_token=None,
        _conversation_id=None,
        _stop_event=asyncio.Event(),
        _interrupt_event=asyncio.Event(),
        _idle_task=None,
        _drain_task=None,
        _audio_queue=asyncio.Queue(),
        _audio_player=object(),
        log_channel=FakeLogChannel(),
        connectionState="connected",
    )
    defaults.update(overrides)
    return FakePC(**defaults)


def make_webpage(asr="google", llm="baidu", tts="azure", vad="silero"):
    """A WebPage instance built without __init__, so no real DB/argparse/network setup runs."""
    wp = server.WebPage.__new__(server.WebPage)
    wp.logger = logging.getLogger("test-server")
    wp.pcs = set()
    wp.args = SimpleNamespace(asr=asr, llm=llm, tts=tts, vad=vad)
    return wp


async def cancel_and_drain(*tasks):
    for task in tasks:
        if task is not None and not task.done():
            task.cancel()
    await asyncio.gather(*(t for t in tasks if t is not None), return_exceptions=True)


class TestActivationGating:
    # A second "activate" on an already-activated connection should just ack, not recharge.
    async def test_noop_if_already_activated(self):
        wp = make_webpage()
        pc = make_pc(_activated=True)

        await wp._activate_session(pc, "PC-1")

        assert pc.log_channel.sent == [{"type": "activate_result", "ok": True, "message": ""}]

    # Activating before the client's audio track has arrived must fail, not crash.
    async def test_rejects_when_no_client_track(self):
        wp = make_webpage()
        pc = make_pc(_client_track=None)

        await wp._activate_session(pc, "PC-2")

        assert pc._activated is False
        assert pc.log_channel.sent[-1]["ok"] is False
        assert "reload" in pc.log_channel.sent[-1]["message"].lower()


class TestActivationQuota:
    # A signed-in user under their limit should activate and be charged exactly once.
    async def test_signed_in_user_success_charges_once(self):
        wp = make_webpage()
        user = auth_store.create_user("alice", "hunter2")
        pc = make_pc(_user=user)

        await wp._activate_session(pc, "PC-3")
        await asyncio.gather(pc._asr_task, pc._llm_task, pc._tts_task)

        assert pc._activated is True
        assert pc._conversation_id is not None
        count, _ = auth_store.get_conversation_usage(user["id"])
        assert count == 1
        activate_results = [m for m in pc.log_channel.sent if m["type"] == "activate_result"]
        assert activate_results == [{"type": "activate_result", "ok": True, "message": ""}]

    # A signed-in user at their limit must be rejected without an extra charge.
    async def test_signed_in_user_over_limit_rejected(self):
        wp = make_webpage()
        user = auth_store.create_user("bob", "hunter2")
        auth_store.set_conversation_limit("bob", 1)
        auth_store.increment_conversation_count(user["id"])  # pre-fill to the limit
        pc = make_pc(_user=user)

        await wp._activate_session(pc, "PC-4")

        assert pc._activated is False
        assert pc._conversation_id is None
        count, limit = auth_store.get_conversation_usage(user["id"])
        assert (count, limit) == (1, 1)
        assert pc.log_channel.sent[-1]["ok"] is False
        assert "limit" in pc.log_channel.sent[-1]["message"].lower()

    # A guest under their trial limit should activate; guest conversations aren't persisted.
    async def test_guest_success_charges_once(self):
        wp = make_webpage()
        guest_token = auth_store.create_guest()
        pc = make_pc(_guest_token=guest_token)

        await wp._activate_session(pc, "PC-5")
        await asyncio.gather(pc._asr_task, pc._llm_task, pc._tts_task)

        assert pc._activated is True
        assert pc._conversation_id is None
        assert auth_store.get_guest_conversation_count(guest_token) == 1

    # A guest who has used up their trial must be rejected without an extra charge.
    async def test_guest_over_limit_rejected(self):
        wp = make_webpage()
        guest_token = auth_store.create_guest()
        for _ in range(auth_store.GUEST_CONVERSATION_LIMIT):
            auth_store.increment_guest_conversation_count(guest_token)
        pc = make_pc(_guest_token=guest_token)

        await wp._activate_session(pc, "PC-6")

        assert pc._activated is False
        assert auth_store.get_guest_conversation_count(guest_token) == auth_store.GUEST_CONVERSATION_LIMIT
        assert pc.log_channel.sent[-1]["ok"] is False

    # An unrecognized/expired guest token should fail cleanly rather than error.
    async def test_guest_unknown_token_rejected(self):
        wp = make_webpage()
        pc = make_pc(_guest_token="never-issued")

        await wp._activate_session(pc, "PC-7")

        assert pc._activated is False
        assert pc.log_channel.sent[-1]["ok"] is False


class TestActivationFailureRollback:
    # If pipeline construction throws after a signed-in user was charged, refund the slot
    # and leave the connection warm (not half-activated) so the client can retry.
    async def test_signed_in_user_failure_refunds_and_reverts_to_warm(self, monkeypatch):
        def failing_create_client(model_type, platform=None, **kwargs):
            if model_type == "asr":
                raise RuntimeError("boom")
            return FakeClient()

        monkeypatch.setattr(server, "create_client", failing_create_client)
        wp = make_webpage()
        user = auth_store.create_user("carol", "hunter2")
        pc = make_pc(_user=user)

        await wp._activate_session(pc, "PC-8")

        assert pc._activated is False
        assert pc._conversation_id is None
        count, _ = auth_store.get_conversation_usage(user["id"])
        assert count == 0  # refunded
        assert pc._drain_task is not None  # re-armed for another attempt
        assert pc._idle_task is not None
        assert pc.log_channel.sent[-1]["ok"] is False
        await cancel_and_drain(pc._drain_task, pc._idle_task)

    # Same rollback guarantee on the guest path.
    async def test_guest_failure_refunds_and_reverts_to_warm(self, monkeypatch):
        def failing_create_client(model_type, platform=None, **kwargs):
            if model_type == "llm":
                raise RuntimeError("boom")
            return FakeClient()

        monkeypatch.setattr(server, "create_client", failing_create_client)
        wp = make_webpage()
        guest_token = auth_store.create_guest()
        pc = make_pc(_guest_token=guest_token)

        await wp._activate_session(pc, "PC-9")

        assert pc._activated is False
        assert auth_store.get_guest_conversation_count(guest_token) == 0  # refunded
        assert pc._drain_task is not None
        assert pc._idle_task is not None
        await cancel_and_drain(pc._drain_task, pc._idle_task)


class TestIdleReap:
    # A warm connection that never activates within the timeout gets closed and discarded.
    async def test_closes_never_activated_connection(self, monkeypatch):
        monkeypatch.setattr(server, "WARM_IDLE_TIMEOUT_S", 0.01)
        wp = make_webpage()
        closed = {"called": False}

        async def fake_close():
            closed["called"] = True

        pc = make_pc(close=fake_close)
        wp.pcs = {pc}

        await wp._close_if_never_activated(pc, "PC-10")

        assert closed["called"] is True
        assert pc not in wp.pcs
        assert pc._idle_task is None  # detached before close, so cleanup can't cancel it mid-close

    # A connection that activated before the timeout must not be closed by the reaper.
    async def test_noop_if_already_activated(self, monkeypatch):
        monkeypatch.setattr(server, "WARM_IDLE_TIMEOUT_S", 0.01)
        wp = make_webpage()
        closed = {"called": False}

        async def fake_close():
            closed["called"] = True

        pc = make_pc(_activated=True, close=fake_close)
        wp.pcs = {pc}

        await wp._close_if_never_activated(pc, "PC-11")

        assert closed["called"] is False
        assert pc in wp.pcs

    # Cancelling the reap task (e.g. because activation just happened) must not close the pc.
    async def test_cancel_before_timeout_does_not_close(self):
        wp = make_webpage()
        closed = {"called": False}

        async def fake_close():
            closed["called"] = True

        pc = make_pc(close=fake_close)
        wp.pcs = {pc}

        task = asyncio.create_task(wp._close_if_never_activated(pc, "PC-12"))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert closed["called"] is False
        assert pc in wp.pcs


class TestDrainTrack:
    # Draining should stop cleanly (no exception escapes) when the track's stream ends.
    async def test_stops_on_stream_error(self):
        wp = make_webpage()

        class EndsImmediately:
            async def recv(self):
                raise server.MediaStreamError

        await wp._drain_track(EndsImmediately())  # must not raise

    # Draining should stop cleanly when cancelled mid-recv (activation taking over the track).
    async def test_stops_cleanly_on_cancel(self):
        wp = make_webpage()
        track = FakeTrack()

        task = asyncio.create_task(wp._drain_track(track))
        await asyncio.sleep(0)  # let it block on track.recv()
        task.cancel()
        await task  # CancelledError is caught inside _drain_track, so this must not raise

        assert task.cancelled() is False
