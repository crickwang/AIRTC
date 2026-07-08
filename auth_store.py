import base64
import hashlib
import logging
import os
import secrets
import sqlite3
from pathlib import Path

from config.settings import ROOT
from supabase import create_client

DB_PATH = Path(ROOT) / "auth.db"
PBKDF2_ITERATIONS = 200_000
SESSION_TTL_SECONDS = 3600
DEFAULT_CONVERSATION_LIMIT = 30

logger = logging.getLogger(__name__)


def backup_to_supabase():
    """
    Upload the current auth.db file to Supabase Storage, overwriting the previous backup.

    No-op if SUPABASE_URL/SUPABASE_KEY aren't set, so this stays silent for anyone
    who hasn't set up Supabase yet. Failures are logged, never raised.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        return

    bucket = os.getenv("SUPABASE_BACKUP_BUCKET", "backups")
    try:
        client = create_client(supabase_url, supabase_key)
        with open(DB_PATH, "rb") as f:
            client.storage.from_(bucket).upload(
                "auth.db", f, {"upsert": "true"}
            )
    except Exception:
        logger.warning("Supabase backup upload failed", exc_info=True)

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_auth_db():
    with _connect() as conn:
        # users table stores user credentials and metadata
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                conversation_count INTEGER NOT NULL DEFAULT 0,
                conversation_limit INTEGER NOT NULL DEFAULT {DEFAULT_CONVERSATION_LIMIT}
            )
            """
        )
        # sessions table stores active user sessions
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
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
            cursor = conn.execute(
                """
                INSERT INTO users (username, password_salt, password_hash, created_at)
                VALUES (?, ?, ?, datetime('now'))
                """,
                (normalized_username, password_salt, password_hash),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists") from exc

        user_id = cursor.lastrowid
    user = get_user_by_id(user_id)
    backup_to_supabase()
    return user


def get_user_by_username(username: str):
    normalized_username = _normalize_username(username)
    with _connect() as conn:
        return conn.execute(
            "SELECT id, username, password_salt, password_hash, created_at FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()


def get_user_by_id(user_id: int):
    with _connect() as conn:
        return conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
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
            INSERT INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, datetime('now'), datetime('now', ?))
            """,
            (token, user_id, f"+{SESSION_TTL_SECONDS} seconds"),
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
            WHERE sessions.token = ? AND sessions.expires_at > datetime('now')
            """,
            (token,),
        ).fetchone()

        if session is None:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

        return session


def get_conversation_usage(user_id: int) -> tuple[int, int]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT conversation_count, conversation_limit FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return (row["conversation_count"], row["conversation_limit"]) if row else (0, 0)


def increment_conversation_count(user_id: int):
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET conversation_count = conversation_count + 1 WHERE id = ?",
            (user_id,),
        )


def set_conversation_limit(username: str, new_limit: int):
    """Override a specific account's conversation limit, e.g. for your own testing account."""
    normalized_username = _normalize_username(username)
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE users SET conversation_limit = ? WHERE username = ?",
            (new_limit, normalized_username),
        )
        if cursor.rowcount == 0:
            raise ValueError("Username not found")
