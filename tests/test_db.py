# AI-generated test suite (Claude) for db.py, written on the `testing` branch.

import sqlite3

import pytest

import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point db at a throwaway sqlite file so tests never touch a real auth.db."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_auth.db")
    monkeypatch.setattr(db, "APP_ENV", "development")
    db.init_auth_db()


def make_user(username="alice", password="hunter2"):
    return db.create_user(username, password)


class TestUsers:
    # A freshly created user should authenticate with the same credentials.
    def test_create_and_authenticate(self):
        make_user()
        user = db.authenticate_user("alice", "hunter2")
        assert user is not None
        assert user["username"] == "alice"

    # A wrong password must not authenticate, even for a real username.
    def test_authenticate_wrong_password_fails(self):
        make_user()
        assert db.authenticate_user("alice", "wrong") is None

    # Authenticating a username that was never created should fail, not error.
    def test_authenticate_unknown_user_fails(self):
        assert db.authenticate_user("nobody", "hunter2") is None

    # Usernames are normalized (trimmed/lowercased), so signup casing/whitespace shouldn't matter.
    def test_username_is_normalized(self):
        make_user(username="  Alice  ")
        user = db.authenticate_user("alice", "hunter2")
        assert user is not None

    # Creating a second user with the same username must raise, not silently overwrite.
    def test_duplicate_username_raises(self):
        make_user()
        with pytest.raises(ValueError):
            make_user()

    # Empty username or password should be rejected before hitting the database.
    def test_create_user_requires_username_and_password(self):
        with pytest.raises(ValueError):
            db.create_user("", "hunter2")
        with pytest.raises(ValueError):
            db.create_user("alice", "")


class TestSessions:
    # A session token just issued for a user should resolve back to that same user.
    def test_create_session_round_trips(self):
        user = make_user()
        token = db.create_session(user["id"])
        session_user = db.get_session_user(token)
        assert session_user is not None
        assert session_user["id"] == user["id"]

    # A token that was never issued should not resolve to any user.
    def test_unknown_token_returns_none(self):
        assert db.get_session_user("not-a-real-token") is None

    # An expired session must be rejected, and the stale row deleted as a side effect (lazy cleanup).
    def test_expired_session_is_rejected_and_cleaned_up(self):
        user = make_user()
        token = db.create_session(user["id"])
        with db._connect() as conn:
            conn.execute(
                "UPDATE sessions SET expires_at = datetime('now', '-1 second') WHERE token = ?",
                (token,),
            )

        assert db.get_session_user(token) is None
        with db._connect() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE token = ?", (token,)
            ).fetchone()[0]
        assert remaining == 0


class TestConversationLimits:
    # A brand-new user should start at zero usage with the documented default limit.
    def test_usage_defaults(self):
        user = make_user()
        count, limit = db.get_conversation_usage(user["id"])
        assert count == 0
        assert limit == db.DEFAULT_CONVERSATION_LIMIT

    # Each call to increment_conversation_count should add exactly one to the running total.
    def test_increment_conversation_count(self):
        user = make_user()
        db.increment_conversation_count(user["id"])
        db.increment_conversation_count(user["id"])
        count, _ = db.get_conversation_usage(user["id"])
        assert count == 2

    # An admin override of a user's limit should be reflected back in get_conversation_usage.
    def test_set_conversation_limit(self):
        make_user()
        db.set_conversation_limit("alice", 5)
        user = db.authenticate_user("alice", "hunter2")
        _, limit = db.get_conversation_usage(user["id"])
        assert limit == 5

    # Setting a limit for a username that doesn't exist should raise, not silently no-op.
    def test_set_conversation_limit_unknown_user_raises(self):
        with pytest.raises(ValueError):
            db.set_conversation_limit("nobody", 5)


class TestConversationsAndMessages:
    # Starting a conversation should return an int id and write a row with started_at set,
    # ended_at still NULL, and the correct owning user_id.
    def test_create_conversation_returns_id_and_row(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        assert isinstance(conversation_id, int)

        with db._connect() as conn:
            row = conn.execute(
                "SELECT user_id, started_at, ended_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        assert row["user_id"] == user["id"]
        assert row["started_at"] is not None
        assert row["ended_at"] is None

    # Messages should be stored in insertion order with their role and content intact —
    # this is what TranscriptLoggingQueue/ResponseLoggingQueue rely on in server.py.
    def test_add_message_persists_role_and_content(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.add_message(conversation_id, "user", "hello there")
        db.add_message(conversation_id, "assistant", "hi, how can I help?")

        with db._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        assert [dict(r) for r in rows] == [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ]

    # add_message should silently skip empty/falsy content instead of writing a blank row.
    def test_add_message_ignores_empty_content(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.add_message(conversation_id, "user", "")

        with db._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
        assert count == 0

    # The messages.role CHECK constraint should reject anything other than user/assistant.
    def test_add_message_rejects_invalid_role(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        with pytest.raises(sqlite3.IntegrityError):
            db.add_message(conversation_id, "system", "not allowed")

    # Ending a conversation should stamp ended_at instead of leaving it NULL.
    def test_end_conversation_sets_ended_at(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.end_conversation(conversation_id)

        with db._connect() as conn:
            ended_at = conn.execute(
                "SELECT ended_at FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()[0]
        assert ended_at is not None

    # end_conversation's WHERE ... AND ended_at IS NULL guard means calling it twice should
    # keep the original end time, not bump it forward on a second call.
    def test_end_conversation_does_not_overwrite_existing_end_time(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.end_conversation(conversation_id)

        with db._connect() as conn:
            first_ended_at = conn.execute(
                "SELECT ended_at FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()[0]

        db.end_conversation(conversation_id)
        with db._connect() as conn:
            second_ended_at = conn.execute(
                "SELECT ended_at FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()[0]
        assert second_ended_at == first_ended_at
