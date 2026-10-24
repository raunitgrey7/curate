# Evaluation methodology

Everything below is implemented in `src/curate/evaluation` and executed by `python -m curate.bench` (or
`make bench`). All numbers in the README and in `REPORT.md` come from `benchmarks/results.json` produced by
that command with `--seed 42`.

## Datasets

### MovieLens 100k

* Source: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`, downloaded once into `data/`.
* Integrity: SHA-256 `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229` (recorded in
  `data/CHECKSUMS.sha256` and in `curate.data.movielens.ML100K_SHA256`; the published MD5
  `0e33842e24a9c977be4e0107933c0723` also matched). A mismatch deletes the file and raises.
* Contents: 100,000 ratings from 943 users on 1,682 movies, September 1997 to April 1998, with user age, gender and
  occupation and 19 genre flags per movie.
* Implicit interpretation: every rating is a "watched" event. Ratings are kept as weights; the counterfactual reward
  on MovieLens is "rated 4 or more in the test window".

### Synthetic retail catalogue

`curate.data.synthetic.generate_retail` produces a seeded implicit-feedback log with known ground truth. Each
assumption has a statistical motivation:

| Mechanism | Choice | Motivation |
|---|---|---|
| Item exposure | Zipf with exponent 1.0 over a random rank | Long-tail sales distributions (Anderson 2006; Clauset, Shalizi and Newman 2009) |
| User activity | Gamma(shape 1.5) rates, Poisson counts | Gamma-Poisson (negative binomial) purchase counts, the NBD model (Ehrenberg 1959) |
| Preferences | Users and items in a 16-dimensional latent space built from 12 category centroids plus noise | Low-rank structure is the assumption behind factorisation models; categories give the side features something to explain |
| Utility | `2.0 * affinity + 0.6 * log popularity` | Exposure bias on top of taste, as in real logs |
| Item choice | Gumbel-top-k over utilities without replacement | Exact sampling from the softmax (Kool, van Hoof and Welling 2019) |
| Timing | Users start at a random day in the first 60 percent of a 180 day horizon; items launch at random days | Gives cold items and realistic time structure for the temporal split |
| Funnel | view -> cart -> purchase with logistic probabilities in utility | Graded implicit weights 1, 2, 3 |
| Reward model | `sigmoid(1.5 * (utility - threshold))`, threshold calibrated so the mean click probability is 5 percent | Exact click probabilities for every (user, item) pair, used by the counterfactual and A/B blocks |

Default size: 2,000 users, 600 items, about 78,000 events. The whole generator runs in well under a second.

## Splits

* **Per-user temporal (primary).** For each user, events are ordered by timestamp; the last 20 percent are test, the
  10 percent before them validation, the rest train (at least one training event is kept). Reason: 68 percent of
  MovieLens 100k users rated everything within one day, so a single global cut leaves about 100 users with history on
  both sides. The known caveat is that other users' later events are visible at training time.
* **Global temporal (secondary).** Two timestamp quantiles cut the whole log into train (80 percent), validation (10
  percent) and test (10 percent). No event in train happens after an event in test. Fewer users are evaluable; the
  numbers are reported alongside.

Evaluable users have at least one training event and at least one test item they had not interacted with before the
test window. Items seen in train or validation are masked before the top-k is taken.

## Protocol for the two-stage model

1. Retrievers are fit on train.
2. Candidates for validation users (top 50 unseen items per retriever, union) are labelled by whether they appear in
   the validation window; a `HistGradientBoostingClassifier` (200 iterations, learning rate 0.05, 31 leaves) is fit on
   them.
3. Retrievers are refit on train plus validation and the trained ranker is applied on their candidates for the test
   window. Every baseline is evaluated with the same train plus validation history, so the only difference between a
   retriever and the two-stage system is the ranking stage.

Hyperparameters were chosen on the validation split only (BPR: dim 128, 50 epochs, lr 0.005, L2 3e-3 after two
sweeps recorded in `REPORT.md`).

## Offline metrics

For a user with relevant set `R` and list `L` of length `k`:

* `Recall@k = |L ∩ R| / |R|` (not capped at `k`; users with 30 relevant items cannot reach 1.0 at `k = 10`).
* `Precision@k = |L ∩ R| / k`.
* `HitRate@k = 1[|L ∩ R| > 0]`.
* `MRR@k = 1 / rank of the first hit`, 0 without a hit.
* `NDCG@k = DCG@k / IDCG@k`, binary gains, discount `1 / log2(rank + 1)`, ideal DCG over `min(|R|, k)`.
* `Coverage@k` = share of catalogue items that appear in at least one evaluated user's list.
* `Novelty@k` = mean over list slots of `-log2 p(i)`, where `p(i) = (users who interacted with i in train + 1) /
  (users + 2)`; higher means less popular recommendations (Zhou et al. 2010; Vargas and Castells 2011).

All user-level metrics are averaged over evaluable users. `k = 10` is the headline; `k = 20` is also in
`results.json`.

## Counterfactual evaluation

Logged bandit feedback is simulated on the test users of the per-user split: a logging policy (softmax over
popularity with temperature 1.0, mixed with 20 percent uniform exploration over unseen items) shows one item per round
for 50,000 rounds; the reward is Bernoulli with the ground-truth probability (synthetic) or the deterministic "liked in
the test window" table (MovieLens). Target policies are "uniform over the model's top-10 unseen items" mixed with 5
percent exploration, for popularity, item-kNN, two-tower and two-stage.

Because the reward table is known, the exact value of every target policy is computed and each estimator's absolute
error and interval coverage is reported:

* IPS: `mean(w_i r_i)`, `w_i = pi(a_i | x_i) / pi_0(a_i | x_i)`.
* IPS with weights clipped at 50.
* SNIPS: `sum(w_i r_i) / sum(w_i)`, standard error from the influence function of the ratio.
* DM: a deliberately simple reward model (per-item logged mean reward shrunk towards the global mean) averaged under
  the target policy.
* DR: DM plus the importance-weighted residual correction (Dudik, Langford and Li 2011).

Effective sample size `(sum w)^2 / sum w^2` is reported for every estimate.

## A/B simulator

Arrivals sample a test user uniformly, assign control (popularity top-10) or treatment (two-stage top-10) with
probability 0.5 and convert with the slate's hit probability `1 - prod(1 - p(u, i))` over the slate (ground truth on
synthetic; deterministic hit on MovieLens). 400 replications of up to 20,000 arrivals are simulated at once and analysed
at every 100 arrivals with:

* mSPRT (Johari, Pekelis and Walsh 2017) with mixing standard deviation `tau = 0.02` and `alpha = 0.05`; a run stops
  the first time the always-valid p-value is at or below alpha;
* a naive z-test applied at every checkpoint (the practice the mSPRT replaces);
* a fixed-horizon z-test at 20,000 arrivals.

Reported: false positive rate under A/A for the three analyses, power (share of replications that declare treatment
better) and mean arrivals at decision under the real lift between the two policies.

## Serving latency

`curate.serving.latency` starts uvicorn in a background thread on a free port and issues 200 sequential
`POST /v1/recommend` requests per model (k = 10, explanations on) through `httpx` after 20 warm-up requests, then 200
requests at concurrency 8 to measure throughput. Percentiles are of wall-clock time per request on the client side.

## Reproducing

```
make setup
make bench            # writes benchmarks/results.json and benchmarks/results.md
make check            # compares against benchmarks/thresholds.json
```

Use `uv run python -m curate.bench --seed 7` to rerun with a different seed; the synthetic generator, the splits, the
models and the simulators all take their randomness from that one seed.
