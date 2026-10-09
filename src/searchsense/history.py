"""The same bounded history policy in memory and in Redis, with strict as-of reads."""
from collections import defaultdict
import hashlib
import json
import redis


def visible(rows, as_of, limit=50):
    return sorted((r for r in rows if int(r[0]) < as_of), key=lambda r: (r[0], r[1]))[-limit:]


class MemoryHistory:
    def __init__(self, limit=50):
        self.rows = defaultdict(list)
        self.limit = limit

    def get(self, user, as_of):
        return visible(self.rows[user], as_of, self.limit)

    def add(self, user, ts, query):
        # Offline callers replay sorted time, preserving ties as invisible to one another.
        self.rows[user].append((int(ts), query))
        self.rows[user] = self.rows[user][-self.limit * 2:]


class RedisHistory:
    def __init__(self, url, namespace="searchsense", limit=50, ttl=86400 * 90):
        self.client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        self.client.ping()
        self.namespace, self.limit, self.ttl = namespace, limit, ttl
        # Atomic add/retention/version update; cache versions cannot become stale after writes.
        self.add_script = self.client.register_script("""
        redis.call('ZADD', KEYS[1], ARGV[1], ARGV[2])
        redis.call('ZREMRANGEBYRANK', KEYS[1], 0, -tonumber(ARGV[3])-1)
        redis.call('EXPIRE', KEYS[1], ARGV[4])
        redis.call('INCR', KEYS[2])
        redis.call('EXPIRE', KEYS[2], ARGV[4])
        return 1
        """)

    def key(self, user):
        return f"{self.namespace}:history:{hashlib.sha256(user.encode()).hexdigest()}"

    def get(self, user, as_of):
        rows = self.client.zrangebyscore(self.key(user), "-inf", f"({int(as_of)}")
        return [json.loads(row) for row in rows[-self.limit:]]

    def add(self, user, ts, query):
        key = self.key(user)
        self.add_script(keys=[key, key + ":version"], args=[int(ts), json.dumps([int(ts), query]), self.limit * 2, self.ttl])

    def version(self, user):
        return self.client.get(self.key(user) + ":version") or "0"

    def cache_get(self, key):
        raw = self.client.get(f"{self.namespace}:cache:{key}")
        return json.loads(raw) if raw else None

    def cache_set(self, key, value):
        self.client.set(f"{self.namespace}:cache:{key}", json.dumps(value), ex=60)
