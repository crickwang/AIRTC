"""
PostgreSQL-backed storage for accounts, sessions, guest trials, and conversation history.

Connection string (DATABASE_URL, TEST_DATABASE_URL) is in .env
"""

import base64
import hashlib
import logging
import os
import secrets

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PBKDF2_ITERATIONS = 200_000
SESSION_TTL_SECONDS = 3600
DEFAULT_CONVERSATION_LIMIT = 30
GUEST_CONVERSATION_LIMIT = 3
GUEST_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

logger = logging.getLogger(__name__)


def _connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set — add it to .env (see README)")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


SCHEMA = f"""
DO $$ BEGIN
    CREATE TYPE message_role AS ENUM ('user', 'assistant');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

/*
users (id PK, username UNIQUE, password_salt, password_hash, created_at, 
       conversation_count, conversation_limit)
sessions (token PK, user_id FK, created_at, expires_at)
    sessions_user_id_idx: index on sessions(user_id)
guests (token PK, created_at, conversation_count)
conversations (id PK, user_id FK, started_at, ended_at)
    conversations_user_started_idx: index on conversations(user_id, started_at DESC)
messages (id PK, conversation_id FK, role, content, created_at)
    messages_conversation_created_idx: index on messages(conversation_id, created_at)
*/

CREATE TABLE IF NOT EXISTS users (
    id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username           TEXT NOT NULL UNIQUE,
    password_salt      TEXT NOT NULL,
    password_hash      TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    conversation_count INTEGER NOT NULL DEFAULT 0 CHECK (conversation_count >= 0),
    conversation_limit INTEGER NOT NULL DEFAULT {DEFAULT_CONVERSATION_LIMIT}
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions (user_id);

CREATE TABLE IF NOT EXISTS guests (
    token              TEXT PRIMARY KEY,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    conversation_count INTEGER NOT NULL DEFAULT 0 CHECK (conversation_count >= 0)
);

CREATE TABLE IF NOT EXISTS conversations (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    BIGINT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at   TIMESTAMPTZ,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS conversations_user_started_idx
    ON conversations (user_id, started_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id BIGINT NOT NULL,
    role            message_role NOT NULL,
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS messages_conversation_created_idx
    ON messages (conversation_id, created_at);
"""


def init_db():
    """Create the schema if it doesn't exist."""
    with _connect() as conn:
        conn.execute(SCHEMA)


def _normalize_username(username: str) -> str:
    """Lowercase and strip whitespace from a username."""
    return username.strip().lower()


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """"""
    if salt is None:
        salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return base64.b64encode(salt).decode("ascii"), base64.b64encode(password_hash).decode("ascii")


def _verify_password(password: str, password_salt: str, password_hash: str) -> bool:
    salt = base64.b64decode(password_salt.encode("ascii"))
    _, computed_hash = _hash_password(password, salt)
    return secrets.compare_digest(computed_hash, password_hash)


def create_user(username: str, password: str):
    normalized_username = _normalize_username(username)
    if not normalized_username or not password:
        raise ValueError("Username and password are required")

    password_salt, password_hash = _hash_password(password)
    with _connect() as conn:
        try:
            return conn.execute(
                """
                INSERT INTO users (username, password_salt, password_hash)
                VALUES (%s, %s, %s)
                RETURNING id, username, created_at
                """,
                (normalized_username, password_salt, password_hash),
            ).fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise ValueError("Username already exists") from exc


def get_user_by_username(username: str):
    normalized_username = _normalize_username(username)
    with _connect() as conn:
        return conn.execute(
            "SELECT id, username, password_salt, password_hash, created_at "
            "FROM users WHERE username = %s",
            (normalized_username,),
        ).fetchone()


def get_user_by_id(user_id: int):
    with _connect() as conn:
        return conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()


def authenticate_user(username: str, password: str):
    user = get_user_by_username(username)
    if user is None:
        return None
    if not _verify_password(password, user["password_salt"], user["password_hash"]):
        return None
    return user


def create_session(user_id: int):
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, expires_at)
            VALUES (%s, %s, now() + make_interval(secs => %s))
            """,
            (token, user_id, SESSION_TTL_SECONDS),
        )
    return token


def get_session_user(token: str):
    if not token:
        return None

    with _connect() as conn:
        session = conn.execute(
            """
            SELECT users.id, users.username, users.created_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = %s AND sessions.expires_at > now()
            """,
            (token,),
        ).fetchone()

        if session is None:
            conn.execute("DELETE FROM sessions WHERE token = %s", (token,))

        return session


def create_guest() -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute("INSERT INTO guests (token) VALUES (%s)", (token,))
    return token


def get_guest_conversation_count(token: str) -> int | None:
    """Returns the guest's usage count, or None if the token is unknown (never issued)."""
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT conversation_count FROM guests WHERE token = %s",
            (token,),
        ).fetchone()
    return row["conversation_count"] if row else None


def try_charge_guest_conversation(token: str) -> bool:
    """
    Atomically consume one guest trial slot. Returns False if the token is unknown or
    the trial is used up. A single conditional UPDATE, so two concurrent activations
    can't both slip past the limit.
    """
    if not token:
        return False
    with _connect() as conn:
        row = conn.execute(
            """
            UPDATE guests SET conversation_count = conversation_count + 1
            WHERE token = %s AND conversation_count < %s
            RETURNING token
            """,
            (token, GUEST_CONVERSATION_LIMIT),
        ).fetchone()
    return row is not None


def increment_guest_conversation_count(token: str):
    with _connect() as conn:
        conn.execute(
            "UPDATE guests SET conversation_count = conversation_count + 1 WHERE token = %s",
            (token,),
        )


def decrement_guest_conversation_count(token: str):
    """Refunds a trial slot when the connection never actually got established."""
    with _connect() as conn:
        conn.execute(
            "UPDATE guests SET conversation_count = conversation_count - 1 "
            "WHERE token = %s AND conversation_count > 0",
            (token,),
        )


def get_conversation_usage(user_id: int) -> tuple[int, int]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT conversation_count, conversation_limit FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
    return (row["conversation_count"], row["conversation_limit"]) if row else (0, 0)


def try_charge_conversation(user_id: int) -> bool:
    """
    Atomically consume one conversation slot. Returns False if the user is at their
    limit. Replaces the old read-check-increment sequence, which let two activations
    racing on the same account both succeed at count == limit - 1.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            UPDATE users SET conversation_count = conversation_count + 1
            WHERE id = %s AND conversation_count < conversation_limit
            RETURNING id
            """,
            (user_id,),
        ).fetchone()
    return row is not None


def increment_conversation_count(user_id: int):
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET conversation_count = conversation_count + 1 WHERE id = %s",
            (user_id,),
        )


def decrement_conversation_count(user_id: int):
    """Refunds a conversation slot when the connection never actually got established."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET conversation_count = conversation_count - 1 "
            "WHERE id = %s AND conversation_count > 0",
            (user_id,),
        )


def set_conversation_limit(username: str, new_limit: int):
    """Override a specific account's conversation limit, e.g. for your own testing account."""
    normalized_username = _normalize_username(username)
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET conversation_limit = %s WHERE username = %s",
            (new_limit, normalized_username),
        )
        if cursor.rowcount == 0:
            raise ValueError("Username not found")


def create_conversation(user_id: int) -> int:
    with _connect() as conn:
        row = conn.execute(
            "INSERT INTO conversations (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        ).fetchone()
    return row["id"]


def end_conversation(conversation_id: int):
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET ended_at = now() WHERE id = %s AND ended_at IS NULL",
            (conversation_id,),
        )


def add_message(conversation_id: int, role: str, content: str):
    if not content:
        return
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, role, content),
        )
