"""Hybrid lexical, semantic and per-user next-word candidates."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import faiss
import numpy as np

FEATURE_NAMES = ["log_global_frequency", "log_prefix_frequency", "log_last_word_frequency",
                 "context_word_cosine", "neural_next_word_logit", "log_history_word_frequency",
                 "log_history_prefix_frequency", "recent_history_weight", "history_context_cosine",
                 "prefix_length", "history_size", "global_recency"]


class Retriever:
    def __init__(self, encoder, stats, index, candidate_limit=64):
        self.encoder, self.stats, self.index = encoder, stats, index
        self.limit = candidate_limit
        self.popular = stats["popular"]

    @classmethod
    def build(cls, encoder, event_factory, directory: Path, candidate_limit=64):
        prefix = defaultdict(Counter)
        last = defaultdict(Counter)
        counts = Counter()
        latest = {}
        for _, ts, query in event_factory():
            words = query.split()
            for word in words:
                if word in encoder.ids:
                    counts[word] += 1
                    latest[word] = max(ts, latest.get(word, 0))
            for i in range(1, len(words)):
                if words[i] in encoder.ids:
                    prefix[" ".join(words[:i])][words[i]] += 1
                    last[words[i - 1]][words[i]] += 1
        stats = {"prefix": {p: dict(c.most_common(32)) for p, c in prefix.items()},
                 "last": {p: dict(c.most_common(64)) for p, c in last.items()},
                 "counts": dict(counts), "latest": latest, "popular": [w for w, _ in counts.most_common(32)],
                 "candidate_limit": candidate_limit}
        # 0/1 are PAD and UNK and excluded from word suggestions.
        vectors = np.ascontiguousarray(encoder.vectors[2:])
        index = faiss.IndexHNSWFlat(vectors.shape[1], 24, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
        index.add(vectors)
        (directory / "retrieval.json").write_text(json.dumps(stats))
        faiss.write_index(index, str(directory / "words.faiss"))
        return cls(encoder, stats, index, candidate_limit)

    @classmethod
    def load(cls, encoder, directory):
        stats = json.loads((directory / "retrieval.json").read_text())
        return cls(encoder, stats, faiss.read_index(str(directory / "words.faiss")), stats["candidate_limit"])

    def retrieve(self, prefix, history):
        words = prefix.split()
        context = self.encoder.encode([prefix])[0]
        unit = context / max(float(np.linalg.norm(context)), 1e-8)
        _, nearest = self.index.search(unit[None, :], min(24, self.index.ntotal))
        lexical = self.stats["prefix"].get(prefix, {})
        suffix = self.stats["last"].get(words[-1] if words else "", {})
        hist_words = Counter(w for _, q in history for w in q.split() if w in self.encoder.ids)
        # Sources have fixed quotas; no true word is inserted during evaluation or training.
        sources = [list(lexical)[:24], list(suffix)[:16],
                   [self.encoder.words[i + 2] for i in nearest[0] if i >= 0],
                   [w for w, _ in hist_words.most_common(12)], self.popular[:12]]
        reserved = [source[:quota] for source, quota in zip(sources, [20, 12, 20, 8, 4])]
        candidates = list(dict.fromkeys(w for source in reserved + sources for w in source))[:self.limit]
        return candidates, context

    def features(self, prefix, candidates, context, history, as_of):
        if not candidates:
            return np.empty((0, len(FEATURE_NAMES)), dtype="float32")
        ids = [self.encoder.ids[w] for w in candidates]
        vectors = self.encoder.vectors[ids]
        unit = context / max(float(np.linalg.norm(context)), 1e-8)
        cosine = vectors @ unit
        logits = self.encoder.output_weights[ids] @ context + self.encoder.output_bias[ids]
        lexical = self.stats["prefix"].get(prefix, {})
        suffix = self.stats["last"].get(prefix.split()[-1] if prefix else "", {})
        hist_words = Counter()
        hist_prefix = Counter()
        recent = defaultdict(float)
        for ts, query in history:
            words = query.split()
            hist_words.update(words)
            age = max(0, as_of - ts) / 86400
            for word in set(words):
                recent[word] += np.exp(-age / 7)
            for i in range(1, len(words)):
                if " ".join(words[:i]) == prefix:
                    hist_prefix[words[i]] += 1
        # Encode history with the same context tower used for the current prefix.
        if history:
            h = self.encoder.encode([q for _, q in history]).mean(0)
            h /= max(float(np.linalg.norm(h)), 1e-8)
            history_sim = vectors @ h
        else:
            history_sim = np.zeros(len(candidates))
        result = []
        for i, word in enumerate(candidates):
            last_seen = self.stats["latest"].get(word, 0)
            result.append([np.log1p(self.stats["counts"].get(word, 0)), np.log1p(lexical.get(word, 0)),
                           np.log1p(suffix.get(word, 0)), cosine[i], logits[i], np.log1p(hist_words[word]),
                           np.log1p(hist_prefix[word]), recent[word], history_sim[i], len(prefix.split()),
                           len(history), np.exp(-max(0, as_of - last_seen) / (86400 * 7))])
        return np.asarray(result, dtype="float32")
