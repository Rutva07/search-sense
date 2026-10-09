import csv
import json
import sqlite3
import pytest
from searchsense.data import normalize, ingest, events, timestamp
from searchsense.history import MemoryHistory


def test_normalization_and_utc():
    assert normalize("  Best   NEW-YORK Hotels! ") == "best new york hotels"
    assert timestamp("2006-03-01 00:00:00") == 1141171200
    assert normalize("www.example.com") == "www example com"


def test_ingestion_deduplicates_clicks_and_orders_globally(tmp_path):
    path = tmp_path / "input.tsv"
    with path.open("w") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["AnonID", "Query", "QueryTime", "ItemRank", "ClickURL"])
        for i in reversed(range(150)):
            row = [str(i%3), "best city hotels", f"2006-03-01 00:{i//60:02d}:{i%60:02d}", 1, "example.org"]
            writer.writerow(row)
            writer.writerow(row[:-2]+[2, "other.org"])
        writer.writerow(["1", "-", "invalid"])
    db = tmp_path / "events.sqlite"
    report = ingest([path], db)
    assert report["raw_rows_read"] == 301
    assert report["deduplicated_events"] == 150
    assert report["users"] == 3
    rows = list(events(db))
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)
    cutoffs = report["cutoffs"]
    assert len(list(events(db, end=cutoffs[0]))) == 75
    with pytest.raises(FileExistsError):
        ingest([path], db)


def test_history_excludes_current_and_future_events():
    history = MemoryHistory(limit=3)
    history.add("u", 10, "past query")
    history.add("u", 20, "tied current query")
    history.add("u", 30, "future query")
    assert history.get("u", 20) == [(10, "past query")]
    assert history.get("unknown", 20) == []
