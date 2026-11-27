# curate

[![ci](https://github.com/raunitgrey7/curate/actions/workflows/ci.yml/badge.svg)](https://github.com/raunitgrey7/curate/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A recommender platform that treats evaluation as the product: five recommenders behind one interface, a
sqlite feature store, time-based offline evaluation, counterfactual (IPS/SNIPS/DM/DR) evaluation of logged
policies against exact ground truth, an A/B simulator with always-valid sequential testing, and a FastAPI
service that returns recommendations with explanations whose numbers are read from the training data.

Everything runs on a laptop CPU with no API key. Every number in this file was produced by
`make bench` with seed 42 and copied from `benchmarks/results.json` by `python -m curate.bench --update-readme`;
`tests/test_docs_consistency.py` fails if the two disagree.

## Why

Most recommender repositories stop at Recall@k on a random split. That number says little about what a
new model will do online. curate adds the three layers a team needs before shipping a ranking change:

1. **Honest offline evaluation.** Per-user and global temporal splits, seen items masked, coverage and novelty
   next to accuracy, and a two-stage protocol in which every baseline sees exactly the same history.
2. **Counterfactual evaluation.** Logged bandit feedback is re-weighted to value a new policy before it is
   deployed. Because the synthetic world has a known reward model, the estimators' errors are measured, not
   assumed.
3. **Sequential A/B testing.** A simulator shows what naive peeking does to the false positive rate and what the
   mSPRT buys in return, with power and time-to-decision measured over hundreds of replications.

## Architecture

```
MovieLens 100k (sha256) ---+                                   +--> offline metrics (Recall, NDCG, coverage, novelty)
                           |   Interactions ---> temporal split |
seeded retail generator ---+        |                           +--> counterfactual (IPS, SNIPS, DM, DR vs truth)
(known reward model)                v                           |
                            sqlite feature store                +--> A/B simulator (mSPRT vs naive vs fixed)
                                    |
             +-----------+----------+-----------+-------------+
             | popularity| item-kNN | BPR (torch)| two-tower  |      retrievers
             +-----------+----------+-----------+-------------+
                          \        |          |        /
                       candidates + per-retriever features
                                    |
                      GBDT two-stage ranker (scikit-learn)
                                    |
          FastAPI: /v1/recommend  /v1/items/{id}/similar  /health  /metrics
          explanations: "Because you watched X: 71% of its viewers also watched this"
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Methodology and metric definitions:
[docs/EVALUATION.md](docs/EVALUATION.md). Every measured number with its command: [REPORT.md](REPORT.md).

## Quickstart

```
make setup     # uv sync, download and verify MovieLens 100k, generate the synthetic catalogue
make test      # pytest
make bench     # train, evaluate, write benchmarks/results.json and results.md (a few minutes on a laptop)
make serve     # trains the serving bundle if missing, then serves on http://127.0.0.1:8000
curl -s -X POST localhost:8000/v1/recommend -H 'content-type: application/json' -d '{"user_id": 1, "k": 5}'
```

Open `http://127.0.0.1:8000/` for a small page that shows recommendations with their explanations, or
`/docs` for the OpenAPI UI. Set `APP_API_KEY` to require an `X-API-Key` header on every route except
`/health` and `/metrics`.

## Measured results

<!-- RESULTS:BEGIN -->
Seed 42, generated 2026-10-24T21:54:36Z, commit `e3a7e7a`, Python 3.13.17, torch 2.14.1, scikit-learn 1.9.1 on arm64. Total benchmark time 150.6 s.

### movielens_100k: per-user temporal split, k = 10 (943 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0666 | 0.1214 | 0.5567 | 0.0428 | 1.395 | 0.0 |
| itemknn | 0.1195 | 0.1932 | 0.6946 | 0.1046 | 1.900 | 0.1 |
| bpr | 0.1168 | 0.1873 | 0.7296 | 0.3686 | 2.452 | 6.9 |
| two_tower | 0.1251 | 0.1998 | 0.7275 | 0.4429 | 2.590 | 6.6 |
| two_stage | 0.1371 | 0.2182 | 0.7784 | 0.3514 | 2.452 | 29.2 |

### retail_synthetic: per-user temporal split, k = 10 (2000 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0968 | 0.0929 | 0.4475 | 0.0533 | 2.671 | 0.0 |
| itemknn | 0.3399 | 0.3637 | 0.8015 | 0.6750 | 3.709 | 0.0 |
| bpr | 0.2980 | 0.3190 | 0.7710 | 0.9483 | 3.918 | 4.3 |
| two_tower | 0.3019 | 0.3215 | 0.7630 | 0.9867 | 4.017 | 4.6 |
| two_stage | 0.3570 | 0.3853 | 0.8085 | 0.8583 | 3.749 | 22.8 |

### Counterfactual estimates of the two-stage top-10 policy

| dataset | true value | IPS (abs error) | SNIPS (abs error) | DR (abs error) | IPS effective sample size |
|---|---|---|---|---|---|
| movielens_100k | 0.1166 | 0.1110 (0.0056) | 0.1176 (0.0010) | 0.1136 (0.0029) | 845 of 50000 |
| retail_synthetic | 0.4561 | 0.4583 (0.0021) | 0.4601 (0.0040) | 0.4551 (0.0010) | 1473 of 50000 |

### A/B simulator, alpha 0.05, 400 replications of up to 20000 arrivals

| dataset | control vs treatment | true lift | A/A mSPRT FPR | A/A naive peeking FPR | A/A fixed-horizon FPR | A/B mSPRT power | mean arrivals at decision |
|---|---|---|---|---|---|---|---|
| movielens_100k | popularity top-10 vs two_stage top-10 | +0.1994 | 0.0075 | 0.4000 | 0.0325 | 1.0000 | 768 |
| movielens_100k | two_tower top-10 vs two_stage top-10 | +0.0456 | 0.0050 | 0.4225 | 0.0400 | 1.0000 | 4966 |
| retail_synthetic | popularity top-10 vs two_stage top-10 | +0.2947 | 0.0150 | 0.4300 | 0.0375 | 1.0000 | 401 |
| retail_synthetic | two_tower top-10 vs two_stage top-10 | +0.0241 | 0.0250 | 0.4000 | 0.0450 | 0.9925 | 6778 |

### Serving latency (movielens_100k, 200 sequential HTTP requests per model, k = 10, explanations on)

| model | p50 ms | p95 ms | p99 ms | req/s at concurrency 8 |
|---|---|---|---|---|
| popularity | 1.16 | 1.39 | 1.48 | 599 |
| itemknn | 1.21 | 1.45 | 1.59 | 569 |
| bpr | 1.18 | 1.41 | 1.78 | 591 |
| two_tower | 1.17 | 1.39 | 1.50 | 594 |
| two_stage | 3.20 | 3.49 | 3.68 | 248 |

Full tables including k = 20, the global temporal split and every estimator: [benchmarks/results.md](benchmarks/results.md).
<!-- RESULTS:END -->

## Models

| Model | What it does | Where it helps |
|---|---|---|
| `popularity` | Interaction counts, optional half-life decay | Floor for every comparison, cold-start fallback |
| `itemknn` | Shrunk cosine item-item similarity, 100 neighbours | Strong on dense catalogues, drives the explanations |
| `bpr` | Matrix factorisation with the BPR pairwise loss in PyTorch | Personalised, broad coverage |
| `two_tower` | Id embeddings plus feature-store side features per tower, in-batch softmax with logQ correction | Uses user and item features, retrieval-ready embeddings |
| `two_stage` | Union of retriever candidates re-ranked by a gradient-boosted classifier | Best accuracy; learns when to trust which retriever |

## API

| Route | Purpose |
|---|---|
| `POST /v1/recommend` | `{"user_id", "k", "model", "exclude_seen", "explain"}`; unknown users get the popularity fallback and `fallback: "popularity_cold_start"` |
| `GET /v1/recommend/{user_id}` | Same as above with query parameters |
| `GET /v1/items/{item_id}` and `/similar` | Catalogue entry and item-kNN neighbours |
| `GET /v1/users/{user_id}/history` | What the model knows about a user |
| `GET /v1/models` | Loaded models and their training times |
| `GET /health`, `GET /metrics` | Liveness payload and Prometheus text format |

Every response carries `X-Request-ID` (propagated when the client sends one) and `X-Process-Time-ms`; logs are
one JSON object per line with the request id. Validation errors return 422 with a `detail` string.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CURATE_DATA_DIR` | `./data` | Dataset cache |
| `CURATE_ARTIFACT_DIR` | `./artifacts` | Serving bundle and feature store |
| `APP_API_KEY` | unset | When set, required as `X-API-Key` |
| `CURATE_SEED` | `42` | Default seed for every command |
| `CURATE_LLM_EXPLAIN` | off | Reword explanations with an Anthropic model; needs `ANTHROPIC_API_KEY` and `uv sync --extra llm` |
| `CURATE_LLM_MODEL` | `claude-opus-5-5` | Model id for the optional rewording (`claude-haiku-5-5` is the cheaper choice) |
| `CURATE_LOG_LEVEL` | `INFO` | Root log level |

## Limitations

* Scoring is a dense pass over the catalogue per user. It is fast at 1,682 or 600 items and fine up to tens of
  thousands; a million-item catalogue needs an ANN index behind the same `score` interface.
* The per-user temporal split lets other users' later events into training. The global split, reported as the
  secondary protocol, does not, but evaluates far fewer MovieLens users because most of them rated everything
  within one day.
* The counterfactual block on MovieLens is semi-synthetic: the logging policy is simulated and the reward is
  "liked in the test window", so unlogged items count as zero reward. The synthetic retail world is the only
  place where the reward model is fully known.
* The A/B simulator assumes independent arrivals and a stationary conversion probability per user and arm; real
  experiments have novelty effects, seasonality and interference.
* The two-stage ranker is a pointwise classifier (no listwise loss) and candidates are capped at 50 per
  retriever.
* No online learning: the bundle is retrained with `curate train`.
* `files.grouplens.org` chains to an eMudhra root that Debian's CA bundle does not ship, so the download fails inside
  `python:3.13-slim` containers. Run `make setup` on the host first; the Docker build copies the verified archive from
  `data/` and only downloads when it is absent.

## Roadmap

* Listwise ranking loss (LambdaRank) and a position-bias model for the ranker.
* Session-based sequence model as a fifth retriever.
* Slate-level off-policy estimators (pseudo-inverse) instead of one-item-per-round logs.
* Streaming feature updates into the sqlite store from the `/v1/feedback` endpoint.

## Development

```
make lint      # ruff check + format check
make test      # pytest, 120+ tests including property-based, adversarial and determinism tests
make check     # quality gate: benchmarks/results.json against benchmarks/thresholds.json
make docker-build && make docker-run
```

CI (`.github/workflows/ci.yml`) runs lint, tests, the benchmark and the quality gate, then builds the image
and smoke-tests `/health` and `/v1/recommend` inside the container.

## License

MIT.
