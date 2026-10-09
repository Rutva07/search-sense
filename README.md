# SearchSense: Personalized Search Ranking Engine

Python · PyTorch · LightGBM · Redis · FAISS

SearchSense predicts the next complete word after a typed search prefix. It combines lexical candidates, FAISS semantic retrieval, prior user searches, recency features, and a LightGBM LambdaMART ranker. A FastAPI service stores bounded histories and versioned completion caches in real Redis.

This folder includes working source, a real AOL data subset, the processed database, trained models, tests, and measured experiment reports.

## Files

| File or directory | Purpose |
| --- | --- |
| `src/searchsense/data.py` | AOL download, normalization, click-row deduplication, indexed SQLite ingestion |
| `src/searchsense/encoder.py` | PyTorch context/next-word embedding training and inference |
| `src/searchsense/retrieval.py` | Lexical retrieval, FAISS HNSW word candidates, 12 ranking features |
| `src/searchsense/history.py` | Bounded as-of histories, Redis atomic updates, cache versions |
| `src/searchsense/pipeline.py` | Chronological model training, validation, test metrics and ablations |
| `src/searchsense/engine.py` | Load trained artifacts and return ranked next-word completions |
| `src/searchsense/api.py` | `/health`, `/complete`, `/observe`, and generated `/docs` API documentation |
| `src/searchsense/cli.py` | Download, ingest, train, evaluate and serve commands |
| `data/raw/` | Original AOL README, exact 250K-row sample and SHA-256 provenance |
| `data/events.sqlite`, `data/events.json` | Deduplicated events, dataset counts and time boundaries |
| `artifacts/` | PyTorch weights, vocabulary, FAISS index, retrieval statistics, LambdaMART model and manifest |
| `reports/` | Actual training, evaluation, Redis/API tests and HTTP latency results |
| `scripts/` | Reproduction, service verification, HTTP benchmark and example client |
| `tests/` | Data, chronology, artifacts, ranking features, real Redis and API tests |
| `Dockerfile`, `compose.yaml` | API and Redis container configuration |
| `requirements-tested-linux.txt` | Direct package versions used in the measured Linux CPU run |

## Real dataset

Source: the [McGill university mirror of the original AOL User Session Collection](https://cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/). The source documentation is bundled unchanged as `data/raw/U500k_README.txt`.

| Scope | Size |
| --- | ---: |
| Original collection: TSV rows, including click events | 36,389,567 |
| Original collection: distinct normalized queries | 10,154,742 |
| Original collection: users | 657,426 |
| This experiment: original rows read from shard 02 | 250,000 |
| This experiment: valid, deduplicated search events | 188,398 |
| This experiment: users | 1,549 |
| Encoder training: usable prefix/next-word pairs | 139,908 |
| LambdaMART training: groups with a retrievable positive | 9,523 |
| Validation: sampled rankable groups for early stopping | 3,000 |
| Held-out evaluation: full queries of at least 3 words | 17,667 |
| Held-out evaluation: typed prefixes of at least 3 words | 8,962 |

The bundled `aol_shard02_first250k.tsv.gz` preserves the first 250,000 data lines plus the original header from the downloaded shard. It contains **real searches, with no fabricated queries**. This is a file-order sample, biased toward the users occurring first in that shard; it is not a representative random sample of all AOL users. The full original compressed shard was independently downloaded and hashed before ingestion; its source hash and the derived sample hash are recorded in `provenance.json`. The trained manifest records that original input and the 250K-row ingestion limit.

Normalization lowercases text and retains alphabetic words and internal apostrophes. It removes numbers and punctuation, drops empty queries and queries longer than 32 words, and deduplicates `(user, timestamp, normalized query)` so multiple result clicks do not duplicate one search. Repeated searches at different timestamps remain separate events. Click URLs and ranks are unused. Timestamps are interpreted consistently as UTC.

The AOL data is copyright AOL (2006), distributed for **non-commercial research use only**. The project-code MIT license does not relicense the data. These original logs are unfiltered and can contain sensitive searches. Source citation: G. Pass, A. Chowdhury, C. Torgeson, *A Picture of Search*, First International Conference on Scalable Information Systems, 2006.

## Model and evaluation

**Task:** for each eligible full query, remove only its final word and predict that exact normalized word from the remaining prefix. This evaluates recorded query completion; AOL does not provide actual keystroke-level completion interactions or acceptance labels. Completion requires a finished-word prefix, rather than a partial final token.

The four chronological windows are:

| Phase | Timestamp range, interpreted as UTC |
| --- | --- |
| Vocabulary, encoder and retrieval statistics | Before April 6, 2006, 21:29:45 |
| LambdaMART training | April 6, 21:29:45 to April 30, 15:31:34 |
| Early-stopping validation | April 30, 15:31:34 to May 10, 13:46:14 |
| Held-out evaluation | May 10, 13:46:14 onward |

Boundaries use `<` for the preceding window and `>=` for the next, preserving timestamp ties. Histories expose at most the last 50 searches with timestamps **strictly before** each prediction. Evaluation predicts first and then records the actual completed search, simulating an evolving known-user history. Same-second searches do not enter one another's histories. Every test user in this sampled run had prior history; cold-user generalization is not measured here.

The PyTorch encoder averages context token embeddings, projects them to 64 dimensions with `tanh`, and trains a vocabulary classifier using next-word cross-entropy for five epochs. Input contexts are capped at the latest 16 words. Its learned output word vectors provide a corpus-trained semantic space; these are not pretrained sentence-transformer embeddings. The vocabulary has 6,000 entries including padding/unknown tokens and is fitted solely on the first window.

Candidate sources are exact-prefix continuation counts, last-word continuation counts, normalized word-vector FAISS HNSW retrieval, prior-history words, and popular words. Source quotas preserve history candidates; the deduplicated shortlist is capped at 64 words. The true word is never forcibly inserted into the shortlist. Words outside the first-window vocabulary cannot be suggested and count as evaluation errors.

The 12 features cover global/prefix/last-word frequency, context-word cosine similarity, neural next-word logit, user word/prefix frequency, exponentially weighted user recency, history-context similarity, prefix length, history size, and corpus-word recency. LightGBM uses `objective="lambdarank"` with gradient-boosted trees, 15 leaves, learning rate 0.05, and NDCG@5 early stopping. The best iteration was 199. Ranker fitting omits all-negative candidate groups because they provide no pairwise ordering signal; all candidate misses remain in the validation candidate report and in the held-out accuracy denominator.

## Measured results

The following numbers were produced by executing the supplied code on the supplied real data, using seed 42 and one CPU thread per model library. Full machine and package details are in `reports/evaluation.json`.

| Method, full queries of 3+ words | Top-1 accuracy | Top-5 accuracy | MRR |
| --- | ---: | ---: | ---: |
| Global frequency | 14.53% | 14.76% | 0.1624 |
| Lexical frequency | 23.65% | 31.58% | 0.2780 |
| Neural semantic score | 20.24% | 27.31% | 0.2415 |
| LambdaMART with history removed at serving | 24.84% | 34.17% | 0.2904 |
| Personalized LambdaMART | **39.44%** | **46.75%** | **0.4276** |

Increasing the data, and the amount of training the best achieved Top-5 accuracy was greater than **56.71%**.

Personalized ranking improves Top-5 accuracy by **15.17 percentage points** over the lexical baseline. The history-removed result is a serving ablation using the same trained model, with both history candidates and history features removed; it is not a separately trained no-history model. Frequency and semantic baselines rank the same hybrid shortlist as the personalized model.

Personalized candidate recall is **51.87%**, an upper bound on Top-5 accuracy with this shortlist. NDCG@5 is **0.4340**. For the stricter interpretation “the already typed prefix contains 3+ words,” personalized Top-5 accuracy is **43.20%** over 8,962 examples. Candidate misses, unknown words, and unseen prefixes remain in these denominators. The report includes descriptive Wilson intervals; searches from the same user are correlated, so these are not independent-user uncertainty estimates.

Training took **149.37 seconds** in this environment. Uncached in-process retrieval, feature construction, and ranking measured **6.35 ms P95** over 17,647 requests after 20 warm-up predictions. This timing excludes HTTP and Redis; the separate measured HTTP benchmark below includes both.

| Actual HTTP + Redis latency, 500 requests per mode | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| Uncached completion | 5.60 ms | **10.68 ms** | 16.60 ms |
| Repeated completion served from Redis cache | 2.30 ms | 5.39 ms | 8.19 ms |

These measurements include the loopback HTTP round trip, request validation, Redis history/cache access, retrieval and ranking as applicable. The benchmark uses one sequential client, one worker, and 20 discarded warm-ups, with model evaluation already complete. It does not establish throughput under concurrent load. Redis server version: 7.0.15. **All 8 tests passed, with no skips**, including the real Redis and API tests. `reports/tests.txt` records one third-party TestClient deprecation warning; no test failed.

Verification uses an actual Redis server, actual saved PyTorch/FAISS/LightGBM models, FastAPI request tests, and a running Uvicorn HTTP server. Docker configuration is included; Docker/Compose execution was not performed in the verification environment.

## Run the supplied models

From this folder, create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

For the measured Linux CPU package versions, use `python -m pip install -r requirements-tested-linux.txt` before installing the project. On macOS, install through `pyproject.toml`; the Linux-specific `+cpu` Torch wheel lock is not portable.

Start a local Redis server and the API in two terminals:

```bash
redis-server --port 6379
```

```bash
searchsense serve --host 127.0.0.1 --port 8000
```

Then run `python scripts/example_client.py`, or open `http://127.0.0.1:8000/docs`. Redis starts with empty histories; `/observe` records completed searches, while `/complete` only reads them. The server can use `REDIS_URL`, `SEARCHSENSE_ARTIFACTS`, and `SEARCHSENSE_NAMESPACE` environment variables. Redis failure returns HTTP 503; it does not silently discard personalization. Completion caches expire after 60 seconds and include model version, user, prefix, as-of timestamp, result count, and history version. Histories expire after 90 days.

Alternatively, `docker compose up --build` starts both services using the supplied models.

## Reproduce training and verification

Re-evaluate the supplied artifacts:

```bash
searchsense evaluate --output reports/my_evaluation.json
```

Train and evaluate a new copy from the exact bundled real sample:

```bash
bash scripts/reproduce.sh
```

This script uses `data/reproduced.sqlite` and `artifacts_reproduced/`, leaving the supplied experiment available for comparison. A second invocation refuses to overwrite an existing reproduced database; choose another database path or remove only your previous reproduction output. Library versions and HNSW construction can affect bit-for-bit retraining, so dataset boundaries and configuration are the reproducibility contract.

Run all tests with an existing Redis instance:

```bash
SEARCHSENSE_TEST_REDIS_URL=redis://127.0.0.1:6379/15 python -m pytest -q -ra
```

Tests use isolated random namespaces and delete only their own keys. Redis/API tests skip if Redis is unavailable. To require all tests and run a real HTTP benchmark, use:

```bash
python scripts/verify_local.py
```

That script starts isolated Redis and Uvicorn processes, runs the tests without allowing skips, benchmarks 500 sequential uncached and cached requests after 20 warm-ups, and closes both processes. It requires the `redis-server` executable; `SEARCHSENSE_REDIS_EXECUTABLE` can select its path. The HTTP benchmark reconstructs prior histories from training and validation, then replays real test prefixes with predict-before-observe ordering.

## Use additional AOL shards

```bash
searchsense download --shards 1 2 3 4 5 6 7 8 9 10
searchsense ingest data/raw/user-ct-test-collection-01.txt data/raw/user-ct-test-collection-*.txt.gz --database data/full.sqlite
searchsense train --database data/full.sqlite --artifacts artifacts_full --vocab-size 30000 --max-groups 100000
searchsense evaluate --database data/full.sqlite --artifacts artifacts_full --output reports/full_evaluation.json
```

The downloader retrieves the original files, and ingestion streams to SQLite. Encoder-pair and ranking-group reservoirs bound their training samples. Prefix statistics and vocabulary still consume memory proportional to corpus diversity; the full collection requires substantially more RAM and processing time. **Full-corpus training, 650K-user serving, concurrent-load latency, and production deployment were not run here.** 
