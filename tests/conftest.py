import os

import pytest

import db


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """
    Point db at the throwaway TEST_DATABASE_URL database and rebuild the schema from
    scratch, so every test starts empty and nothing touches the real DATABASE_URL.
    """
    test_url = os.getenv("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL is not set")
    monkeypatch.setattr(db, "DATABASE_URL", test_url)
    with db._connect() as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    db.init_db()
