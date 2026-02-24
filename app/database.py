"""SQLite persistence for count events using aiosqlite."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

import aiosqlite

from .config import settings

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS count_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    lane        INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,
    class_name  TEXT    NOT NULL
);
"""


async def init_db(db_path: Optional[str] = None) -> None:
    path = db_path or settings.db_path
    async with aiosqlite.connect(path) as db:
        await db.execute(_CREATE_TABLE)
        await db.commit()


async def insert_event(
    lane: int,
    track_id: int,
    class_name: str,
    db_path: Optional[str] = None,
) -> None:
    path = db_path or settings.db_path
    ts = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT INTO count_events (ts, lane, track_id, class_name) VALUES (?, ?, ?, ?)",
            (ts, lane, track_id, class_name),
        )
        await db.commit()


async def get_total_counts(
    db_path: Optional[str] = None,
) -> dict:
    """Return {lane1: int, lane2: int, total: int} from DB."""
    path = db_path or settings.db_path
    async with aiosqlite.connect(path) as db:
        cursor = await db.execute(
            "SELECT lane, COUNT(*) FROM count_events GROUP BY lane"
        )
        rows = await cursor.fetchall()
    counts = {1: 0, 2: 0}
    for lane, n in rows:
        counts[lane] = n
    return {
        "lane1": counts[1],
        "lane2": counts[2],
        "total": counts[1] + counts[2],
    }


def get_total_counts_sync(db_path: Optional[str] = None) -> dict:
    """Synchronous version used in tests or startup."""
    path = db_path or settings.db_path
    try:
        con = sqlite3.connect(path)
        cur = con.execute("SELECT lane, COUNT(*) FROM count_events GROUP BY lane")
        rows = cur.fetchall()
        con.close()
    except Exception:
        rows = []
    counts = {1: 0, 2: 0}
    for lane, n in rows:
        counts[lane] = n
    return {
        "lane1": counts[1],
        "lane2": counts[2],
        "total": counts[1] + counts[2],
    }
