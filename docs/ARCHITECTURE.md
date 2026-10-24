# Architecture

curate is a single Python package (`src/curate`) with five layers that only depend downwards: data, features,
models, evaluation, serving. The benchmark and the CLI sit on top and are the only places that wire layers
together.

```
                 +-----------------------+        +------------------------------+
  MovieLens zip  |  data/movielens.py    |        |  data/synthetic.py           |
  (sha256 check) |  download, verify,    |        |  seeded retail generator     |
  ------------>  |  parse to Dataset     |        |  Dataset + RetailWorld truth |
                 +-----------+-----------+        +---------------+--------------+
                             |                                    |
                             v                                    v
                 +---------------------------------------------------------------+
                 |  data/interactions.py: Interactions (user, item, ts, weight)  |
                 |  data/split.py: per-user temporal | global temporal           |
                 +-------------------------------+-------------------------------+
                                                 |
                 +-------------------------------v-------------------------------+
                 |  features/build.py -> features/store.py (sqlite)              |
                 |  item: genres, decade, log popularity   user: profile,        |
                 |  log activity, genre affinity          metadata: titles       |
                 +-------------------------------+-------------------------------+
                                                 |  FitContext
      +------------------+------------------+----v-------------+------------------+
      | popularity.py    | itemknn.py       | mf.py (BPR)      | two_tower.py     |
      | counts, decay    | shrunk cosine    | PyTorch pairwise | PyTorch in-batch |
      +------------------+------------------+------------------+------------------+
                 \                 |                 |                /
                  \   top-N candidates + per-retriever score, rank  /
                   +----------------------v---------------------------+
                   | ranker.py: TwoStageRecommender (HistGradientBoosting) |
                   +----------------------+---------------------------+
                                          |
          +-------------------------------+--------------------------------+
          |                               |                                |
 evaluation/offline.py          evaluation/counterfactual.py     evaluation/ab.py
 Recall, NDCG, coverage,        IPS, SNIPS, DM, DR vs exact      mSPRT vs naive peeking
 novelty on the test window     policy value on logged bandit    vs fixed horizon, A/A
                                feedback                         and A/B replications
          |                               |                                |
          +-------------------------------+--------------------------------+
                                          |
                            bench.py -> benchmarks/results.json, results.md
                                          |
                      pipeline.py: fit_for_serving -> artifacts/bundle.pkl + features.sqlite
                                          |
                 serving/app.py (FastAPI) + serving/state.py + explain/engine.py
                 /health  /metrics  /v1/recommend  /v1/items/{id}/similar  /v1/users/{id}/history
```

## Modules

| Module | Responsibility | Why it is shaped this way |
|---|---|---|
| `data/interactions.py` | Frozen containers in a contiguous index space with validation in `__post_init__` | Every model works on arrays and sparse matrices; validating once at the boundary removes checks everywhere else |
| `data/movielens.py` | Download to a temp file, verify SHA-256, extract only the needed members, parse latin-1 files | Checksum before anything is cached; zip members with `..` are skipped |
| `data/synthetic.py` | Zipf exposure, Gamma-Poisson activity, low-rank preferences, Gumbel-top-k choice, purchase funnel | Known latent utility makes the counterfactual estimators and the A/B simulator testable against exact values |
| `data/split.py` | Per-user temporal split (primary) and global temporal split (secondary) | 68 percent of MovieLens users rate everything in one day, so a global cut evaluates very few users |
| `features/store.py` | sqlite tables for feature sets (float32 blobs), feature names and JSON metadata | One file ships inside the container; point lookups on the request path; a matrix read is one query |
| `features/build.py` | Train-only item and user features; writes and reads back through the store | The store is on the training path, not a decoration |
| `models/base.py` | `Recommender` protocol, `FitContext`, `topk_from_scores` | Masking, ranking and tie-breaking are implemented once |
| `models/popularity.py` | Count-based baseline with optional half-life decay | The floor every model must beat and the cold-start fallback |
| `models/itemknn.py` | Shrunk cosine item-item similarity, top-N neighbours, kept co-occurrence counts | Co-occurrence counts feed the "because you watched" explanations |
| `models/mf.py` | BPR matrix factorisation in PyTorch, per-epoch negative sampling | Per-epoch sampling cut fit time from 25 s to under 5 s |
| `models/two_tower.py` | Id embedding + side features per tower, in-batch softmax with logQ correction | Retrieval model that can use the feature store; L2-normalised outputs give cosine scores |
| `models/ranker.py` | Candidate union from retrievers, per-user standardised scores and ranks, GBDT classifier | Learns when to trust which retriever; single-threaded GBDT because two OpenMP runtimes contend |
| `evaluation/metrics.py` | Recall, NDCG, precision, hit rate, MRR, coverage, novelty | Definitions are spelled out in `docs/EVALUATION.md` |
| `evaluation/offline.py` | Evaluable users, relevance matrix, batched scoring, exclusion of seen items | Shared across every model so the comparison is identical |
| `evaluation/counterfactual.py` | `BanditFeedback`, IPS, SNIPS, DM, DR, exact policy value, log simulation | Every estimate carries a standard error and an effective sample size |
| `evaluation/policies.py` | Softmax, top-k and uniform policies over unseen items with exploration | Full support makes importance weights finite |
| `evaluation/ab.py` | Vectorised A/B replications, mSPRT, naive peeking, fixed-horizon z-test | Measures false positive rate and power instead of asserting them |
| `pipeline.py` | Fit retrievers on train, ranker on validation, refit retrievers on more data, bundle for serving | Every model sees the same history at test time |
| `serving/*` | FastAPI app, ASGI middleware (request id, JSON access log, metrics, API key), Pydantic schemas, latency harness | Middleware is plain ASGI so it costs microseconds and works with streaming |
| `explain/engine.py` | kNN anchor with measured support, embedding fallback, popularity fallback | Explanations quote numbers read from the fitted models |
| `explain/llm.py` | Optional rewording with the Anthropic SDK behind `CURATE_LLM_EXPLAIN` | Never a dependency: any failure returns the deterministic text |
| `bench.py` | Runs every block with one seed, writes `results.json` and `results.md`, `--check` is the CI gate | The README table is copied from this output |

## Data flow for one request

1. `RequestContextMiddleware` reads or creates `X-Request-ID`, starts a timer and stores the id in a context variable
   used by the JSON log formatter.
2. `ApiKeyMiddleware` rejects the request with 401 unless the path is `/health` or `/metrics` or the key matches.
3. The route validates the body with `RecommendRequest` (unknown fields, negative ids and `k` outside 1 to 100
   are 422s).
4. `ServingState.recommend` maps the external user id to a row, scores all items with the requested model (two-stage by
   default), masks the user's history, takes the top-k and asks the `Explainer` for each item.
5. The response carries the request id, the model time, and for each item the title, genres, year, score and
   explanation. The middleware adds `X-Request-ID` and `X-Process-Time-ms` headers, logs one JSON line and updates
   `curate_http_requests_total` and `curate_http_request_duration_seconds`.

## Training flow (`curate train` and `curate.bench`)

1. Split interactions per user in time: train, validation, test.
2. Build the feature store from the training window.
3. Fit the four retrievers on train; generate candidates for validation users; fit the GBDT on validation labels.
4. Refit the retrievers on train plus validation (benchmark) or on everything (serving); attach the trained ranker.
5. Persist `artifacts/bundle.pkl`, `artifacts/features.sqlite` and `artifacts/manifest.json`.

## Operational notes

* Determinism: numpy generators and `torch.manual_seed` are seeded from one `--seed`; `torch.set_num_threads` is
  capped so results do not depend on the machine's core count.
* Scale: scoring is a dense pass over the catalogue per user, fine up to tens of thousands of items. Beyond that the
  two-tower item embeddings would go into an ANN index behind the same `score` interface.
* Container: multi-stage image; the trainer stage downloads and verifies MovieLens and trains the bundle so the runtime
  stage needs no network and runs as a non-root user.
