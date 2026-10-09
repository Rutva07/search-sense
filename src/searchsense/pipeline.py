"""Chronological training/evaluation; every truth is exposed only after prediction."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import time
import importlib.metadata
import numpy as np
import lightgbm as lgb
from .data import events
from .encoder import train_encoder
from .engine import SearchEngine, configure_threads
from .history import MemoryHistory
from .retrieval import Retriever, FEATURE_NAMES


def build_groups(database, start, end, retriever, max_groups, seed):
    history = MemoryHistory()
    groups = []
    rng = np.random.default_rng(seed)
    seen = eligible = misses = 0
    for user, ts, query in events(database, end=end):
        words = query.split()
        if ts >= start and len(words) >= 3:
            eligible += 1
            prefix, target = " ".join(words[:-1]), words[-1]
            past = history.get(user, ts)
            candidates, context = retriever.retrieve(prefix, past)
            if target not in candidates:
                misses += 1
            else:
                features = retriever.features(prefix, candidates, context, past, ts)
                labels = np.asarray([int(w == target) for w in candidates], dtype="int32")
                group = (features, labels)
                seen += 1
                if len(groups) < max_groups:
                    groups.append(group)
                else:
                    k = int(rng.integers(seen))
                    if k < max_groups:
                        groups[k] = group
        history.add(user, ts, query)
    if not groups:
        raise ValueError("No rankable groups; use more data")
    return (np.concatenate([g[0] for g in groups]), np.concatenate([g[1] for g in groups]),
            [len(g[1]) for g in groups]), {"eligible_queries": eligible, "candidate_misses": misses,
                                         "rankable_groups": seen, "sampled_groups": len(groups)}


def train(database: Path, directory: Path, epochs=5, vocab_size=8000, max_groups=12000, seed=42):
    configure_threads()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset = json.loads(database.with_suffix(".json").read_text())
    c0, c1, c2 = dataset["cutoffs"]
    encoder, encoder_stats = train_encoder(lambda: events(database, end=c0), directory,
                                           epochs=epochs, vocab_size=vocab_size, seed=seed)
    retriever = Retriever.build(encoder, lambda: events(database, end=c0), directory)
    train_data, train_stats = build_groups(database, c0, c1, retriever, max_groups, seed)
    validation_data, validation_stats = build_groups(database, c1, c2, retriever, min(3000, max_groups), seed)
    print(json.dumps({"ranker_training": train_stats, "ranker_validation": validation_stats}), flush=True)
    ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=200,
                          num_leaves=15, learning_rate=0.05, min_child_samples=40,
                          reg_lambda=1, random_state=seed, n_jobs=1, verbosity=-1,
                          deterministic=True, force_col_wise=True)
    ranker.fit(train_data[0], train_data[1], group=train_data[2],
               eval_set=[(validation_data[0], validation_data[1])], eval_group=[validation_data[2]],
               eval_at=[5], feature_name=FEATURE_NAMES,
               callbacks=[lgb.early_stopping(20, verbose=False)])
    ranker.booster_.save_model(str(directory / "ranker.txt"))
    report = {"seed": seed, "dataset": dataset, "encoder": encoder_stats,
              "ranker_train": train_stats, "ranker_validation": validation_stats,
              "best_iteration": ranker.best_iteration_, "features": FEATURE_NAMES,
              "feature_importance_gain": dict(zip(FEATURE_NAMES, ranker.booster_.feature_importance("gain").tolist())),
              "training_seconds": time.perf_counter() - started,
              "split_policy": "retrieval/encoder <c0; LambdaMART c0<=t<c1; validation c1<=t<c2; test t>=c2",
              "history_policy": "strictly earlier timestamps; bounded last 50 searches; predict then observe",
              "artifact_sha256": {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                                  for p in directory.iterdir() if p.suffix in {".pt", ".faiss", ".txt", ".json"} and p.name != "model_manifest.json"}}
    (directory / "model_manifest.json").write_text(json.dumps(report, indent=2))
    return report


def summarize(ranks, coverages):
    ranks = np.asarray(ranks, dtype="float64")
    n = len(ranks)
    if n == 0:
        return {"queries": 0, "top1": None, "top5": None, "mrr": None, "ndcg5": None, "candidate_recall": None}
    hits = ranks <= 5
    accuracy = float(hits.mean())
    # Wilson interval for a Bernoulli hit indicator (descriptive; users are correlated).
    z = 1.96
    denom = 1 + z*z/n
    center = (accuracy + z*z/(2*n))/denom
    half = z*np.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/denom
    return {"queries": n, "top1": float((ranks == 1).mean()), "top5": accuracy,
            "mrr": float(np.where(np.isfinite(ranks), 1/ranks, 0).mean()),
            "ndcg5": float(np.where(hits, 1/np.log2(ranks+1), 0).mean()),
            "candidate_recall": float(np.mean(coverages)), "top5_wilson95": [center-half, center+half]}


def evaluate(database: Path, directory: Path, output: Path):
    engine = SearchEngine(directory)
    history = MemoryHistory()
    dataset = json.loads(database.with_suffix(".json").read_text())
    test_start = dataset["cutoffs"][2]
    methods = ["global_frequency", "lexical_frequency", "semantic", "lambdamart_no_history", "personalized_lambdamart"]
    ranks = {name: [] for name in methods}
    long_ranks = {name: [] for name in methods}
    coverage, long_coverage, latencies = [], [], []
    known = cold = 0
    for user, ts, query in events(database):
        words = query.split()
        if ts >= test_start and len(words) >= 3:
            prefix, target = " ".join(words[:-1]), words[-1]
            past = history.get(user, ts)
            known += bool(past)
            cold += not bool(past)
            started = time.perf_counter_ns()
            candidates, context = engine.retriever.retrieve(prefix, past)
            features = engine.retriever.features(prefix, candidates, context, past, ts)
            personalized = engine.model.predict(features, num_threads=1)
            latencies.append((time.perf_counter_ns()-started)/1e6)
            # Remove history from both retrieval and features for a serving ablation.
            no_candidates, no_context = engine.retriever.retrieve(prefix, [])
            no_features = engine.retriever.features(prefix, no_candidates, no_context, [], ts)
            no_scores = engine.model.predict(no_features, num_threads=1)
            scores = {"global_frequency": (candidates, features[:, 0]),
                      "lexical_frequency": (candidates, 100*features[:, 1] + 10*features[:, 2] + features[:, 0]),
                      "semantic": (candidates, features[:, 4]),
                      "lambdamart_no_history": (no_candidates, no_scores),
                      "personalized_lambdamart": (candidates, personalized)}
            covered = target in candidates
            coverage.append(covered)
            if len(words) >= 4:
                long_coverage.append(covered)
            for name, (choices, values) in scores.items():
                order = np.argsort(-np.asarray(values), kind="stable")
                ordered = [choices[i] for i in order]
                rank = ordered.index(target)+1 if target in ordered else float("inf")
                ranks[name].append(rank)
                if len(words) >= 4:
                    long_ranks[name].append(rank)
        history.add(user, ts, query)
    latency = np.asarray(latencies[20:])  # declared warmup; excludes no-history ablation time
    report = {"task": "Last-word prediction: remove the final word of each full query of >=3 words; exact normalized token match",
              "all_test_queries_in_denominator": True, "misses_count_as_incorrect": True,
              "test_start_utc": datetime.fromtimestamp(test_start, timezone.utc).isoformat(),
              "full_query_3plus_words": {name: summarize(ranks[name], coverage if name != "lambdamart_no_history" else [np.isfinite(r) for r in ranks[name]]) for name in methods},
              "typed_prefix_3plus_words": {name: summarize(long_ranks[name], long_coverage if name != "lambdamart_no_history" else [np.isfinite(r) for r in long_ranks[name]]) for name in methods},
              "test_known_history": known, "test_empty_history": cold,
              "engine_latency_ms": {"samples": len(latency), "warmup": 20, "p50": float(np.percentile(latency, 50)),
                                    "p95": float(np.percentile(latency, 95)), "p99": float(np.percentile(latency, 99)),
                                    "scope": "uncached in-process retrieval+history feature computation+LightGBM; excludes HTTP and Redis"},
              "machine": {"platform": platform.platform(), "processor": platform.processor(), "threads": 1},
              "dependencies": {p: importlib.metadata.version(p) for p in ["torch", "lightgbm", "faiss-cpu", "redis", "fastapi", "numpy"]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report
