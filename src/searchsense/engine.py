"""Artifact loading, serving and model-versioned Redis completion cache."""
import hashlib
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import torch
import faiss
from .data import normalize
from .encoder import Encoder
from .history import MemoryHistory, RedisHistory
from .retrieval import Retriever


def configure_threads():
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)


class SearchEngine:
    def __init__(self, directory: Path, history=None):
        configure_threads()
        self.encoder = Encoder.load(directory)
        self.retriever = Retriever.load(self.encoder, directory)
        self.model = lgb.Booster(model_file=str(directory / "ranker.txt"))
        self.history = history if history is not None else MemoryHistory()
        manifest = directory / "model_manifest.json"
        self.version = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else hashlib.sha256((directory / "ranker.txt").read_bytes()).hexdigest()

    def complete(self, user, prefix, as_of, k=5, use_cache=True):
        prefix = normalize(prefix)
        if not prefix or len(prefix.split()) > 32:
            raise ValueError("Prefix must contain 1 to 32 words")
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        cached = isinstance(self.history, RedisHistory) and use_cache
        if cached:
            version = self.history.version(user)
            key = hashlib.sha256(json.dumps([self.version, user, prefix, int(as_of), k, version]).encode()).hexdigest()
            hit = self.history.cache_get(key)
            if hit is not None:
                return {**hit, "cache_hit": True}
        history = self.history.get(user, int(as_of))
        candidates, context = self.retriever.retrieve(prefix, history)
        features = self.retriever.features(prefix, candidates, context, history, int(as_of))
        scores = self.model.predict(features, num_threads=1) if len(candidates) else []
        order = np.argsort(-np.asarray(scores), kind="stable")[:k]
        result = {"prefix": prefix, "suggestions": [{"word": candidates[i], "completion": prefix + " " + candidates[i],
                   "score": float(scores[i])} for i in order], "candidate_count": len(candidates),
                   "history_size": len(history), "cache_hit": False}
        if cached:
            self.history.cache_set(key, result)
        return result

    def observe(self, user, query, ts):
        query = normalize(query)
        if not query or len(query.split()) > 32:
            raise ValueError("Query must contain 1 to 32 words")
        self.history.add(user, int(ts), query)
