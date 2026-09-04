# AI-generated test suite (Claude) for db.py, originally written on the `testing` branch
# and ported to PostgreSQL on the `database` branch. The isolated_db fixture in
# tests/conftest.py rebuilds the schema in TEST_DATABASE_URL before every test.

import psycopg
import pytest

import db


def make_user(username="alice", password="hunter2"):
    return db.create_user(username, password)


def query_one(sql, params=()):
    with db._connect() as conn:
        return conn.execute(sql, params).fetchone()


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

    # create_user returns the new row directly (INSERT ... RETURNING), so the id it
    # hands back must match what a lookup finds.
    def test_create_user_returns_persisted_row(self):
        user = make_user()
        assert db.get_user_by_id(user["id"])["username"] == "alice"


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
                "UPDATE sessions SET expires_at = now() - interval '1 second' WHERE token = %s",
                (token,),
            )

        assert db.get_session_user(token) is None
        remaining = query_one("SELECT COUNT(*) AS n FROM sessions WHERE token = %s", (token,))
        assert remaining["n"] == 0


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

    # try_charge_conversation succeeds exactly `limit` times, then refuses without
    # touching the count — the check and the increment are one statement.
    def test_try_charge_conversation_stops_at_limit(self):
        user = make_user()
        db.set_conversation_limit("alice", 2)
        assert db.try_charge_conversation(user["id"]) is True
        assert db.try_charge_conversation(user["id"]) is True
        assert db.try_charge_conversation(user["id"]) is False
        count, _ = db.get_conversation_usage(user["id"])
        assert count == 2

    # A refund after a failed activation should re-open a slot.
    def test_decrement_reopens_slot(self):
        user = make_user()
        db.set_conversation_limit("alice", 1)
        assert db.try_charge_conversation(user["id"]) is True
        db.decrement_conversation_count(user["id"])
        assert db.try_charge_conversation(user["id"]) is True


class TestGuests:
    # A token that was never issued has no usage row, which the caller treats as "unknown".
    def test_unknown_guest_returns_none(self):
        assert db.get_guest_conversation_count("nope") is None

    # A fresh guest starts at zero and can be charged GUEST_CONVERSATION_LIMIT times.
    def test_try_charge_guest_stops_at_limit(self):
        token = db.create_guest()
        assert db.get_guest_conversation_count(token) == 0
        for _ in range(db.GUEST_CONVERSATION_LIMIT):
            assert db.try_charge_guest_conversation(token) is True
        assert db.try_charge_guest_conversation(token) is False
        assert db.get_guest_conversation_count(token) == db.GUEST_CONVERSATION_LIMIT

    # Charging an unknown token must fail rather than creating a row.
    def test_try_charge_unknown_guest_fails(self):
        assert db.try_charge_guest_conversation("nope") is False
        assert db.get_guest_conversation_count("nope") is None

    # Refunding below zero is clamped by the WHERE guard, and the CHECK backs it up.
    def test_decrement_guest_never_goes_negative(self):
        token = db.create_guest()
        db.decrement_guest_conversation_count(token)
        assert db.get_guest_conversation_count(token) == 0


class TestConversationsAndMessages:
    # Starting a conversation should return an int id and write a row with started_at set,
    # ended_at still NULL, and the correct owning user_id.
    def test_create_conversation_returns_id_and_row(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        assert isinstance(conversation_id, int)

        row = query_one(
            "SELECT user_id, started_at, ended_at FROM conversations WHERE id = %s",
            (conversation_id,),
        )
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
                "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id",
                (conversation_id,),
            ).fetchall()
        assert rows == [
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "hi, how can I help?"},
        ]

    # add_message should silently skip empty/falsy content instead of writing a blank row.
    def test_add_message_ignores_empty_content(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.add_message(conversation_id, "user", "")

        row = query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = %s", (conversation_id,)
        )
        assert row["n"] == 0

    # messages.role is an enum, so anything other than user/assistant is rejected by Postgres.
    def test_add_message_rejects_invalid_role(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        with pytest.raises(psycopg.DataError):
            db.add_message(conversation_id, "system", "not allowed")

    # Deleting a user should cascade through conversations to messages.
    def test_deleting_user_cascades(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.add_message(conversation_id, "user", "hello")
        with db._connect() as conn:
            conn.execute("DELETE FROM users WHERE id = %s", (user["id"],))
        assert query_one("SELECT COUNT(*) AS n FROM conversations")["n"] == 0
        assert query_one("SELECT COUNT(*) AS n FROM messages")["n"] == 0

    # Ending a conversation should stamp ended_at instead of leaving it NULL.
    def test_end_conversation_sets_ended_at(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.end_conversation(conversation_id)

        row = query_one("SELECT ended_at FROM conversations WHERE id = %s", (conversation_id,))
        assert row["ended_at"] is not None

    # end_conversation's WHERE ... AND ended_at IS NULL guard means calling it twice should
    # keep the original end time, not bump it forward on a second call.
    def test_end_conversation_does_not_overwrite_existing_end_time(self):
        user = make_user()
        conversation_id = db.create_conversation(user["id"])
        db.end_conversation(conversation_id)
        first = query_one("SELECT ended_at FROM conversations WHERE id = %s", (conversation_id,))

        db.end_conversation(conversation_id)
        second = query_one("SELECT ended_at FROM conversations WHERE id = %s", (conversation_id,))
        assert second["ended_at"] == first["ended_at"]
