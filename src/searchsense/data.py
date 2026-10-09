"""Stream original AOL TSV logs into a deduplicated, chronological SQLite store."""
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MIRROR = "https://cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/"


def normalize(query: str) -> str:
    return " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", query.lower()))


def timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def download(output: Path, shards: list[int]) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard in shards:
        if not 1 <= shard <= 10:
            raise ValueError("Shard must be between 1 and 10")
        name = f"user-ct-test-collection-{shard:02d}.txt" + ("" if shard == 1 else ".gz")
        target = output / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(MIRROR + name, timeout=120) as source, temporary.open("wb") as sink:
                while block := source.read(1024 * 1024):
                    sink.write(block)
            temporary.replace(target)
        paths.append(target)
    return paths


def ingest(paths: list[Path], database: Path, max_rows: int | None = None) -> dict:
    if database.exists():
        raise FileExistsError(f"Refusing to append to existing database: {database}")
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE events(user TEXT, ts INTEGER, query TEXT, PRIMARY KEY(user, ts, query)) WITHOUT ROWID")
    read = invalid = rows = 0
    manifest = []
    try:
        for path in paths:
            digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            manifest.append({"file": path.name, "sha256": digest, "bytes": path.stat().st_size})
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as source:
                reader = csv.DictReader(source, delimiter="\t")
                if not {"AnonID", "Query", "QueryTime"}.issubset(reader.fieldnames or []):
                    raise ValueError(f"Invalid AOL header: {path}")
                batch = []
                for row in reader:
                    if max_rows is not None and read >= max_rows:
                        break
                    read += 1
                    try:
                        query = normalize(row["Query"] or "")
                        ts = timestamp(row["QueryTime"])
                        user = row["AnonID"]
                        if not query or not user or len(query.split()) > 32:
                            invalid += 1
                            continue
                        batch.append((user, ts, query))
                    except (ValueError, TypeError, KeyError):
                        invalid += 1
                        continue
                    if len(batch) >= 10000:
                        conn.executemany("INSERT OR IGNORE INTO events VALUES(?,?,?)", batch)
                        conn.commit()
                        batch.clear()
                conn.executemany("INSERT OR IGNORE INTO events VALUES(?,?,?)", batch)
                conn.commit()
            if max_rows is not None and read >= max_rows:
                break
        conn.execute("CREATE INDEX chronological ON events(ts, user, query)")
        rows, users, first, last = conn.execute("SELECT count(*), count(DISTINCT user), min(ts), max(ts) FROM events").fetchone()
        if rows < 100:
            raise ValueError("At least 100 valid unique events required")
        # Timestamp boundaries prevent a tied second from straddling two phases.
        cuts = [conn.execute("SELECT ts FROM events ORDER BY ts LIMIT 1 OFFSET ?", (int(rows * f),)).fetchone()[0]
                for f in (0.5, 0.7, 0.8)]
        if not first < cuts[0] < cuts[1] < cuts[2] < last:
            raise ValueError("Insufficient distinct timestamps for chronological splits")
        stats = {"source": MIRROR, "inputs": manifest, "raw_rows_read": read, "invalid_rows": invalid,
                 "deduplicated_events": rows, "users": users, "first_timestamp": first,
                 "last_timestamp": last, "cutoffs": cuts, "max_rows": max_rows,
                 "sampling": "first N raw rows in file order (not a random or representative sample)" if max_rows else "all supplied rows"}
        database.with_suffix(".json").write_text(json.dumps(stats, indent=2))
        return stats
    finally:
        conn.close()


def events(database: Path, start: int | None = None, end: int | None = None):
    conn = sqlite3.connect(database)
    clauses, values = [], []
    if start is not None:
        clauses.append("ts >= ?")
        values.append(start)
    if end is not None:
        clauses.append("ts < ?")
        values.append(end)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    try:
        yield from conn.execute("SELECT user, ts, query FROM events" + where + " ORDER BY ts, user, query", values)
    finally:
        conn.close()
