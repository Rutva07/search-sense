from pathlib import Path
import json
import hashlib
import numpy as np
from searchsense.engine import SearchEngine
from searchsense.pipeline import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_actual_artifact_hashes_and_heldout_boundary():
    manifest = json.loads((ROOT / "artifacts/model_manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        with (ROOT / "artifacts" / name).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == expected
    assert sorted(manifest["dataset"]["cutoffs"]) == manifest["dataset"]["cutoffs"]


def test_encoder_faiss_ranker_roundtrip_and_personal_features():
    engine = SearchEngine(ROOT / "artifacts")
    user, ts = "test-only-user", 1148000000
    prefix = "best hotels in"
    candidates, context = engine.retriever.retrieve(prefix, [])
    assert len(candidates) == len(set(candidates))
    assert all(w not in {"<unk>", "<pad>"} for w in candidates)
    chosen = candidates[0]
    before = engine.retriever.features(prefix, candidates, context, [], ts)
    engine.observe(user, prefix + " " + chosen, ts-100)
    history = engine.history.get(user, ts)
    after = engine.retriever.features(prefix, candidates, context, history, ts)
    assert after[0, 5] > before[0, 5]
    assert after[0, 6] > before[0, 6]
    result = engine.complete(user, prefix, ts)
    assert len(result["suggestions"]) == 5
    assert all(item["completion"].startswith(prefix + " ") for item in result["suggestions"])
    assert np.isfinite([item["score"] for item in result["suggestions"]]).all()
    assert engine.complete(user, prefix, ts) == result


def test_metrics_keep_candidate_misses_in_denominator():
    result = summarize([1, 5, 6, float("inf")], [True, True, True, False])
    assert result["queries"] == 4
    assert result["top1"] == .25
    assert result["top5"] == .5
    assert result["candidate_recall"] == .75
    assert np.isclose(result["mrr"], (1 + 1/5 + 1/6)/4)
