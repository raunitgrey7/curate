# REPORT: curate

What was built, how every published number was produced, and what is still missing. Companion documents:
`README.md` (overview and results), `docs/ARCHITECTURE.md` (modules and data flow), `docs/EVALUATION.md`
(methodology and metric definitions).

## 1. What was built

A recommender platform in one Python 3.13 package (`src/curate`, managed with `uv`):

| Area | Delivered |
|---|---|
| Data | MovieLens 100k loader with SHA-256 verification and cached extraction; seeded synthetic retail generator with a known reward model; per-user and global temporal splits |
| Feature store | sqlite store (`features/store.py`) holding float32 feature vectors per entity and JSON metadata; builders compute train-only user and item features and read them back through the store |
| Models | popularity (optional decay), item-kNN (shrunk cosine), BPR matrix factorisation (PyTorch), two-tower retriever (PyTorch, in-batch softmax with logQ correction), two-stage GBDT re-ranker over the union of retriever candidates |
| Offline evaluation | Recall, NDCG, precision, hit rate, MRR, coverage, novelty at k = 10 and 20 on the test window with seen items masked |
| Counterfactual evaluation | IPS, clipped IPS, SNIPS, DM and DR with standard errors and effective sample size, compared against exact policy values on simulated logs |
| A/B simulator | Vectorised replications with mSPRT always-valid p-values, naive peeking and fixed-horizon z-tests; A/A false positive rate, power and arrivals-to-decision |
| Serving | FastAPI with Pydantic schemas, request ids, JSON access logs, Prometheus metrics, optional API key, popularity fallback for unknown users, explanations with measured co-occurrence support, a vanilla JS page at `/` |
| Explanations | "Because you watched X: N percent of the people who watched it also watched this; both are Drama, Romance", with an embedding fallback and a popularity fallback; optional Anthropic rewording behind a flag |
| Benchmark | `python -m curate.bench` writes `benchmarks/results.json` and `results.md`; `--check` enforces `benchmarks/thresholds.json`; `--update-readme` copies the tables into the README |
| Packaging | Makefile (`setup lint test bench check train serve docker-build`), multi-stage Dockerfile with a non-root runtime stage, GitHub Actions workflow with the benchmark as a quality gate |
| Tests | 126 tests (unit, integration, property-based with Hypothesis, adversarial, determinism, README-to-results consistency) |

## 2. Design decisions worth recording

* **Per-user temporal split as the primary protocol.** 68 percent of MovieLens 100k users rated everything within
  one day (measured during data inspection: median user span 0.03 days), so a global time cut leaves 69 evaluable
  users at the 80/10/10 quantiles. The per-user split keeps all 943 users evaluable; the global split is reported
  as the leakage-free secondary protocol.
* **Two-stage protocol.** Retrievers are fit on train, the ranker on validation-window labels, then the retrievers
  are refit on train plus validation and reused with the trained ranker. Every baseline is evaluated with the same
  train plus validation history, so the comparison isolates the ranking stage.
* **BPR hyperparameters were chosen on the validation window only.** Two sweeps on MovieLens (validation
  recall@10, never test):
  * Sweep 1 (dim in 32/64, epochs 10/20, lr 0.005/0.01, L2 1e-3/1e-2/5e-2): best dim 64, 20 epochs, lr 0.005,
    L2 1e-3 at 0.1153.
  * Sweep 2 (dim 64/128, epochs 30/50, lr 0.003/0.005, L2 1e-4/1e-3/3e-3): best dim 128, 50 epochs, lr 0.005,
    L2 3e-3 at 0.1297; dim 64 with the same settings reached 0.1236.
  The first BPR implementation sampled negatives per batch and took 25 s per fit; sampling once per epoch cut this
  to under 5 s with the same loss curve.
* **Single-threaded GBDT.** With PyTorch loaded in the same process, scikit-learn's OpenMP pool contended with
  PyTorch's: the ranker fit took 38 s with the default thread count, 11.4 s with 4 threads and 1.1 s with 1 thread
  on a 20k-row training table. `threadpoolctl` pins the fit and predict to one thread.
* **Counterfactual estimators work on distinct contexts.** A first version evaluated policies for all 50,000
  logged rows and materialised 50,000 by 1,682 matrices repeatedly; evaluating the 943 distinct users once and
  indexing rows cut the block from minutes to 0.3 s.
* **mSPRT in log space.** The likelihood ratio overflows `exp` for large lifts; the p-value is computed from the log
  ratio instead.
* **Deterministic first.** No network at runtime, no API key needed; the LLM rewording of explanations is behind
  `CURATE_LLM_EXPLAIN` and returns the deterministic sentence on any failure.

## 3. Measured results

All numbers come from `make bench` (seed 42) on this machine (Apple Silicon, CPU only, Python 3.13.17, torch
2.14.1, scikit-learn 1.9.1). The full tables are in `benchmarks/results.md`; `benchmarks/results.json` is the
source of truth and `tests/test_docs_consistency.py` checks the README against it.

<!-- MEASURED:BEGIN -->
Source: `benchmarks/results.md` (Benchmark results (seed 42, 2026-10-24T21:54:36Z, commit e3a7e7a)).

### movielens_100k: per_user_temporal (943 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0666 | 0.1214 | 0.5567 | 0.0428 | 1.395 | 0.0 |
| itemknn | 0.1195 | 0.1932 | 0.6946 | 0.1046 | 1.900 | 0.1 |
| bpr | 0.1168 | 0.1873 | 0.7296 | 0.3686 | 2.452 | 6.9 |
| two_tower | 0.1251 | 0.1998 | 0.7275 | 0.4429 | 2.590 | 6.6 |
| two_stage | 0.1371 | 0.2182 | 0.7784 | 0.3514 | 2.452 | 29.2 |

### movielens_100k: global_temporal (69 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0644 | 0.1202 | 0.4058 | 0.0470 | 1.626 | 0.0 |
| itemknn | 0.0391 | 0.1142 | 0.2464 | 0.0820 | 2.060 | 0.1 |
| bpr | 0.0704 | 0.1432 | 0.3913 | 0.1718 | 2.460 | 5.9 |
| two_tower | 0.0803 | 0.1505 | 0.4493 | 0.2099 | 2.829 | 5.8 |
| two_stage | 0.1013 | 0.1619 | 0.4783 | 0.1457 | 2.357 | 24.7 |

### retail_synthetic: per_user_temporal (2000 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0968 | 0.0929 | 0.4475 | 0.0533 | 2.671 | 0.0 |
| itemknn | 0.3399 | 0.3637 | 0.8015 | 0.6750 | 3.709 | 0.0 |
| bpr | 0.2980 | 0.3190 | 0.7710 | 0.9483 | 3.918 | 4.3 |
| two_tower | 0.3019 | 0.3215 | 0.7630 | 0.9867 | 4.017 | 4.6 |
| two_stage | 0.3570 | 0.3853 | 0.8085 | 0.8583 | 3.749 | 22.8 |

### retail_synthetic: global_temporal (1698 evaluated users)

| model | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | novelty@10 | fit s |
|---|---|---|---|---|---|---|
| popularity | 0.0953 | 0.0730 | 0.3086 | 0.0617 | 2.500 | 0.0 |
| itemknn | 0.3744 | 0.3267 | 0.7103 | 0.7517 | 3.588 | 0.0 |
| bpr | 0.3357 | 0.2908 | 0.6779 | 0.9600 | 3.774 | 5.6 |
| two_tower | 0.3460 | 0.2964 | 0.6843 | 0.9917 | 3.881 | 3.9 |
| two_stage | 0.3951 | 0.3521 | 0.7291 | 0.8933 | 3.633 | 24.6 |

### Counterfactual evaluation (top-10 target policies, 5 percent exploration)

#### movielens_100k (50000 logged rounds; reward: 1 if the user rated the item 4 or more in the test window, else 0)

| policy | true value | IPS | SNIPS | DM | DR | IPS ESS |
|---|---|---|---|---|---|---|
| popularity | 0.0708 | 0.0639 (err 0.0069) | 0.0668 (err 0.0040) | 0.0528 (err 0.0180) | 0.0664 (err 0.0045) | 1446 |
| itemknn | 0.1108 | 0.1031 (err 0.0078) | 0.1124 (err 0.0015) | 0.0442 (err 0.0667) | 0.1076 (err 0.0032) | 1122 |
| two_tower | 0.1128 | 0.1070 (err 0.0058) | 0.1159 (err 0.0031) | 0.0305 (err 0.0824) | 0.1087 (err 0.0042) | 789 |
| two_stage | 0.1166 | 0.1110 (err 0.0056) | 0.1176 (err 0.0010) | 0.0350 (err 0.0816) | 0.1136 (err 0.0029) | 845 |

#### retail_synthetic (50000 logged rounds; reward: exact click probability from the generator's latent utility)

| policy | true value | IPS | SNIPS | DM | DR | IPS ESS |
|---|---|---|---|---|---|---|
| popularity | 0.1074 | 0.1081 (err 0.0007) | 0.1086 (err 0.0012) | 0.1073 (err 0.0001) | 0.1085 (err 0.0011) | 5582 |
| itemknn | 0.4465 | 0.4473 (err 0.0008) | 0.4445 (err 0.0020) | 0.0623 (err 0.3842) | 0.4438 (err 0.0027) | 1546 |
| two_tower | 0.4131 | 0.4004 (err 0.0127) | 0.4004 (err 0.0127) | 0.0528 (err 0.3603) | 0.3973 (err 0.0158) | 1078 |
| two_stage | 0.4561 | 0.4583 (err 0.0021) | 0.4601 (err 0.0040) | 0.0593 (err 0.3968) | 0.4551 (err 0.0010) | 1473 |

### A/B simulator (alpha 0.05, 400 replications, up to 20000 arrivals, checks every 100)

| dataset | scenario | control hit rate | treatment hit rate | A/A mSPRT FPR | A/A naive peeking FPR | A/A fixed-horizon FPR | A/B mSPRT power | A/B fixed-horizon power | A/B mean n at decision |
|---|---|---|---|---|---|---|---|---|---|
| movielens_100k | popularity top-10 vs two_stage top-10 | 0.4263 | 0.6257 | 0.0075 | 0.4000 | 0.0325 | 1.0000 | 1.0000 | 768 |
| movielens_100k | two_tower top-10 vs two_stage top-10 | 0.5801 | 0.6257 | 0.0050 | 0.4225 | 0.0400 | 1.0000 | 1.0000 | 4966 |
| retail_synthetic | popularity top-10 vs two_stage top-10 | 0.5999 | 0.8947 | 0.0150 | 0.4300 | 0.0375 | 1.0000 | 1.0000 | 401 |
| retail_synthetic | two_tower top-10 vs two_stage top-10 | 0.8706 | 0.8947 | 0.0250 | 0.4000 | 0.0450 | 0.9925 | 1.0000 | 6778 |

### Serving latency (movielens_100k, 200 sequential requests per model, k=10 with explanations)

| model | p50 ms | p95 ms | p99 ms | mean ms | req/s at concurrency 8 |
|---|---|---|---|---|---|
| popularity | 1.16 | 1.39 | 1.48 | 1.18 | 599 |
| itemknn | 1.21 | 1.45 | 1.59 | 1.24 | 569 |
| bpr | 1.18 | 1.41 | 1.78 | 1.21 | 591 |
| two_tower | 1.17 | 1.39 | 1.50 | 1.22 | 594 |
| two_stage | 3.20 | 3.49 | 3.68 | 3.21 | 248 |

Total benchmark time: 150.6 s.
<!-- MEASURED:END -->

## 4. Commands and output tails

<!-- OUTPUT:BEGIN -->
Run from a fresh login shell with `zsh -lc` on 2026-10-25, in this order (the benchmark runs first so the README
consistency tests see the freshly generated tables):

```
make bench && make lint && make test && make check
```

Outcome: `make bench` and `make lint` passed on the first pass; `make test` failed one test
(`test_readme_latency_matches_results_json`: its regex matched the offline table row before the latency row, a bug
in the new consistency test, not in the code under test). After scoping the regex to the latency section, `make test`
and `make check` were rerun from a fresh login shell and passed (`exit=0`). The tails below are from the passing runs.

### `make bench` (tail)

```
uv run python -m curate.bench --seed 42
[bench] movielens_100k: offline, counterfactual and A/B blocks done (62s)
[bench] retail_synthetic: offline, counterfactual and A/B blocks done (121s)
[bench] wrote benchmarks/results.json and benchmarks/results.md in 150.6s
uv run python -m curate.bench --update-readme
README results section updated
```

### `make lint` (tail)

```
uv run ruff check src tests
All checks passed!
uv run ruff format --check src tests
53 files already formatted
```

### `make test` (tail)

```
uv run pytest -q
........................................................................ [ 57%]
......................................................                   [100%]
126 passed in 3.37s
```

### `make check` (tail)

```
uv run python -m curate.bench --check
PASS offline.movielens_100k.per_user_temporal.models.two_stage.recall@10 = 0.1371 (want >= 0.12)
PASS offline.movielens_100k.per_user_temporal.models.two_stage.ndcg@10 = 0.2182 (want >= 0.19)
PASS offline.movielens_100k.per_user_temporal.models.itemknn.recall@10 = 0.1195 (want >= 0.1)
PASS offline.movielens_100k.per_user_temporal.models.two_tower.recall@10 = 0.1251 (want >= 0.1)
PASS offline.movielens_100k.per_user_temporal.models.bpr.recall@10 = 0.1168 (want >= 0.09)
PASS offline.movielens_100k.per_user_temporal.models.popularity.recall@10 = 0.0666 (want >= 0.05)
PASS offline.movielens_100k.per_user_temporal.models.two_stage.coverage@10 = 0.3514 (want >= 0.2)
PASS offline.retail_synthetic.per_user_temporal.models.two_stage.recall@10 = 0.3570 (want >= 0.3)
PASS offline.retail_synthetic.per_user_temporal.models.itemknn.recall@10 = 0.3399 (want >= 0.28)
PASS offline.retail_synthetic.per_user_temporal.models.two_tower.recall@10 = 0.3019 (want >= 0.24)
PASS offline.retail_synthetic.per_user_temporal.models.bpr.recall@10 = 0.2980 (want >= 0.24)
PASS offline.retail_synthetic.per_user_temporal.models.popularity.recall@10 = 0.0968 (want >= 0.06)
PASS counterfactual.retail_synthetic.policies.two_stage.snips.abs_error = 0.0040 (want <= 0.02)
PASS counterfactual.retail_synthetic.policies.two_stage.dr.abs_error = 0.0010 (want <= 0.02)
PASS counterfactual.movielens_100k.policies.two_stage.snips.abs_error = 0.0010 (want <= 0.02)
PASS ab_simulator.retail_synthetic.large_lift.aa.msprt_reject_rate = 0.0150 (want <= 0.08)
PASS ab_simulator.retail_synthetic.small_lift.aa.msprt_reject_rate = 0.0250 (want <= 0.08)
PASS ab_simulator.retail_synthetic.large_lift.aa.naive_peeking_reject_rate = 0.4300 (want >= 0.2)
PASS ab_simulator.retail_synthetic.large_lift.ab.msprt_treatment_better_rate = 1.0000 (want >= 0.9)
PASS ab_simulator.movielens_100k.large_lift.aa.msprt_reject_rate = 0.0075 (want <= 0.08)
PASS serving.models.two_stage.p95_ms = 3.4940 (want <= 50.0)
PASS serving.models.itemknn.p95_ms = 1.4510 (want <= 25.0)
quality gate: PASS
exit=0
```
<!-- OUTPUT:END -->

## 5. From-scratch verification

A fresh `git clone` of the repository into an empty directory, followed by the Makefile targets, on 2026-10-25:

```
=== make setup ===
uv run python -m curate.cli data --seed 42
movielens_100k: 100000 interactions, sha256 verified
retail_synthetic: 77910 interactions (seed 42)
=== make lint ===
All checks passed!
53 files already formatted
=== make test ===
126 passed, 1 warning in 33.15s
=== make check (uses committed results.json) ===
quality gate: PASS
```

The one warning was Starlette's deprecation notice about `httpx` in its test client when `httpx2` is absent; the
main environment had `httpx2` through the optional `anthropic` extra, so `httpx2` was added to the dev group to
make a fresh `uv sync` identical.

## 6. `make serve` from a fresh login shell

`zsh -lc 'make serve PORT=18080'` loaded the bundle trained by the benchmark's latency block and answered the probes
below; the server was then stopped with SIGTERM (make reports that as `Error 143`).

```
GET /health
{"status":"ok","version":"0.1.0","dataset":"movielens_100k","models":["bpr","itemknn","popularity","two_stage","two_tower"],"n_users":943,"n_items":1682,"uptime_seconds":0.63}

POST /v1/recommend {"user_id": 1, "k": 3}
two_stage, fallback none, model time 21.43 ms (first request after start; the warm p50 in the benchmark is 3.20 ms)
  1 Die Hard: With a Vengeance (1995) | Because you watched "Die Hard 2 (1990)": 77% of the people who watched it also watched this one; both are Action, Thriller.
  2 True Lies (1994) | Because you watched "Jurassic Park (1993)": 66% of the people who watched it also watched this one; both are Action, Adventure.
  3 Grease (1978) | Because you watched "Sleepless in Seattle (1993)": 58% of the people who watched it also watched this one; both are Comedy, Romance.

GET /metrics (excerpt)
curate_recommendations_total{model="two_stage"} 1.0

GET /  ->  the static dashboard (<!doctype html> ...)
```

## 7. Container

<!-- DOCKER:BEGIN -->
`docker build -t curate:local .` built the three-stage image on 2026-10-25 (builder: `python:3.13-slim-bookworm`
plus the `uv` 0.13.5 binary; trainer: copies the cached, verified archive and runs `curate data` and `curate train`;
runtime: non-root user, only the virtualenv and `artifacts/`). `docker image inspect` reports 1576 MB on disk
(344 MB compressed content); the size is dominated by the CPU torch wheel and its dependencies.

Smoke test of the running container (`docker run -p 18000:8000 curate:local`):

```
GET /health
{"status":"ok","version":"0.1.0","dataset":"movielens_100k","models":["bpr","itemknn","popularity","two_stage","two_tower"],"n_users":943,"n_items":1682,"uptime_seconds":0.567}

POST /v1/recommend {"user_id": 1, "k": 3}  (excerpt)
{"user_id":1,"model":"two_stage","k":3,"fallback":"none","items":[{"item_id":403,"title":"Batman (1989)", ...
 "explanation":{"kind":"because_you_watched","text":"Because you watched \"Top Gun (1986)\": 70% of the people who watched it also watched this one; both are Action.", ...

docker exec curate-smoke id
uid=999(curate) gid=999(curate) groups=999(curate)

GET /metrics (excerpt)
curate_http_requests_total{method="GET",route="/health",status="200"} 2.0
curate_http_requests_total{method="POST",route="/v1/recommend",status="200"} 1.0
```

Two build problems were found and fixed on the way: the combined `uv` plus Python base image tag did not exist (the
builder now copies the `uv` binary into `python:3.13-slim-bookworm`), and the first runtime command passed
`--log-config /dev/null` to uvicorn, which it rejects as an empty file; the container now starts through
`python -m curate.cli serve`, the same entry point as `make serve`.
<!-- DOCKER:END -->

## 8. Known gaps and limitations

* **Scale.** Scoring is a dense pass over the catalogue per user. Fine for thousands of items; a production
  catalogue of millions needs an ANN index for the two-tower embeddings and a sparse path for item-kNN.
* **Semi-synthetic counterfactual block on MovieLens.** The logging policy is simulated and the reward is "liked in
  the test window", so unlogged items count as zero reward. Only the synthetic retail world has a fully known reward
  model, which is where the estimator errors are exact.
* **Direct method is deliberately weak.** The reward model is a per-item mean shrunk towards the global mean; it is
  there to show how DR recovers from a poor model, not as a competitive estimator.
* **A/B simulator assumptions.** Independent arrivals, stationary per-user conversion probabilities, two arms with
  equal allocation. Novelty effects, seasonality and interference are not modelled. With the two policies measured
  here the large-lift scenario is decided within a few hundred arrivals; the small-lift scenario (two-tower vs
  two-stage) is the informative one.
* **Pointwise ranker.** The two-stage model uses a binary classifier; a listwise loss and a position-bias model are
  on the roadmap. Candidates are capped at 50 per retriever.
* **Global temporal split on MovieLens evaluates only 69 users**, so those numbers carry wide uncertainty and are
  reported for transparency, not as the headline.
* **Latency was measured over loopback with one uvicorn worker** on a laptop; it includes HTTP, middleware,
  explanations and JSON serialisation but not network distance or multi-worker contention.
* **The published numbers come from macOS.** CI on GitHub Actions runs lint, the tests, the benchmark and its
  quality gate on Linux on every push, but the tables in this file were measured on the laptop named above.
* **The MovieLens download does not verify inside Debian-based images.** `files.grouplens.org` chains to the
  "emSign Root TLS CA - G3" root (eMudhra), which the `python:3.13-slim-bookworm` CA bundle does not include (PyPI
  verifies fine from the same container, so this is not a proxy issue; macOS trusts the root, so `make setup` works
  on the host). The Docker build therefore copies the cached `data/ml-100k.zip` from the build context and only
  downloads when it is absent; CI restores the archive from the Actions cache before building. The verification of
  the SHA-256 is unchanged.
* **No online learning or feedback endpoint**; the bundle is retrained with `curate train`.
