"""Measure real HTTP + Redis latency with chronological AOL test-prefix replay."""
import argparse
import json
from pathlib import Path
import platform
import time
import httpx
import numpy as np
from searchsense.data import events
from searchsense.history import RedisHistory, MemoryHistory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--namespace", default="searchsense")
    parser.add_argument("--database", type=Path, default=Path("data/events.sqlite"))
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("reports/http_latency.json"))
    args = parser.parse_args()
    store = RedisHistory(args.redis_url, args.namespace)
    # Reconstruct bounded history at test start in memory, then seed Redis once.
    history = MemoryHistory()
    start = json.loads(args.database.with_suffix(".json").read_text())["cutoffs"][2]
    for user, ts, query in events(args.database, end=start):
        history.add(user, ts, query)
    for user, rows in history.rows.items():
        for ts, query in rows:
            store.add(user, ts, query)
    timings = {"uncached": [], "cached": []}
    seen = 0
    with httpx.Client(base_url=args.url, timeout=15, trust_env=False) as client:
        client.get("/health").raise_for_status()
        for user, ts, query in events(args.database, start=start):
            words = query.split()
            if len(words) >= 3:
                payload = {"user_id": user, "prefix": " ".join(words[:-1]), "timestamp": ts, "use_cache": False}
                t0 = time.perf_counter_ns()
                response = client.post("/complete", json=payload)
                response.raise_for_status()
                uncached = (time.perf_counter_ns()-t0)/1e6
                assert len(response.json()["suggestions"]) == 5
                payload["use_cache"] = True
                client.post("/complete", json=payload).raise_for_status()  # populate
                t0 = time.perf_counter_ns()
                response = client.post("/complete", json=payload)
                response.raise_for_status()
                cached = (time.perf_counter_ns()-t0)/1e6
                assert response.json()["cache_hit"]
                if seen >= 20:
                    timings["uncached"].append(uncached)
                    timings["cached"].append(cached)
                seen += 1
            client.post("/observe", json={"user_id": user, "query": query, "timestamp": ts}).raise_for_status()
            if seen >= args.requests + 20:
                break
    report = {"scope": "HTTP round trip, request validation, Redis history/cache, FAISS, feature computation, LambdaMART; loopback, single sequential client",
              "warmup": 20, "platform": platform.platform(), "workers": 1, "concurrency": 1,
              "redis_version": store.client.info("server")["redis_version"],
              "measurements_ms": {key: {"samples": len(values), "p50": float(np.percentile(values, 50)),
                                        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
                                        "max": float(max(values))} for key, values in timings.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
