"""
One-off: copy the old SQLite auth database into Postgres.

    python scripts/migrate_sqlite_to_postgres.py /path/to/auth.db [--database-url URL]

Reads users, guests, conversations and messages from the SQLite file and inserts them
into the Postgres database at DATABASE_URL (or --database-url), keeping the original
ids so foreign keys still line up, then bumps each identity sequence past the highest
copied id. Sessions are not copied: they expire within an hour and users just log in
again.

Refuses to run if the target already has users, so it can't be applied twice by
accident. Wraps everything in one transaction: either every row lands or none do.

SQLite stored timestamps as naive 'YYYY-MM-DD HH:MM:SS' strings produced by
datetime('now'), which is UTC, so they are loaded as UTC.
"""

import argparse
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db  # noqa: E402


def parse_ts(value):
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sqlite_path")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL is not set and --database-url was not given")
    if not Path(args.sqlite_path).is_file():
        sys.exit(f"SQLite file not found: {args.sqlite_path}")

    src = sqlite3.connect(args.sqlite_path)
    src.row_factory = sqlite3.Row

    db.DATABASE_URL = args.database_url
    db.init_db()

    with db._connect() as pg:
        existing = pg.execute("SELECT count(*) AS n FROM users").fetchone()["n"]
        if existing:
            sys.exit(f"Target already has {existing} users; refusing to import on top of them")

        users = src.execute("SELECT * FROM users").fetchall()
        pg.cursor().executemany(
            """
            INSERT INTO users (id, username, password_salt, password_hash, created_at,
                               conversation_count, conversation_limit)
            OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    u["id"],
                    u["username"],
                    u["password_salt"],
                    u["password_hash"],
                    parse_ts(u["created_at"]),
                    u["conversation_count"],
                    u["conversation_limit"],
                )
                for u in users
            ],
        )

        guests = src.execute("SELECT * FROM guests").fetchall()
        pg.cursor().executemany(
            "INSERT INTO guests (token, created_at, conversation_count) VALUES (%s, %s, %s)",
            [(g["token"], parse_ts(g["created_at"]), g["conversation_count"]) for g in guests],
        )

        conversations = src.execute("SELECT * FROM conversations").fetchall()
        pg.cursor().executemany(
            """
            INSERT INTO conversations (id, user_id, started_at, ended_at)
            OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s, %s)
            """,
            [
                (c["id"], c["user_id"], parse_ts(c["started_at"]), parse_ts(c["ended_at"]))
                for c in conversations
            ],
        )

        messages = src.execute("SELECT * FROM messages").fetchall()
        pg.cursor().executemany(
            """
            INSERT INTO messages (id, conversation_id, role, content, created_at)
            OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (m["id"], m["conversation_id"], m["role"], m["content"], parse_ts(m["created_at"]))
                for m in messages
            ],
        )

        # Identity sequences don't notice explicit ids; move each past the highest one copied.
        for table in ("users", "conversations", "messages"):
            pg.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"COALESCE((SELECT max(id) FROM {table}), 0) + 1, false)"
            )

    print(
        f"copied {len(users)} users, {len(guests)} guests, "
        f"{len(conversations)} conversations, {len(messages)} messages"
    )


if __name__ == "__main__":
    main()
