"""Benchmark harness: ``python -m curate.bench`` trains, evaluates and writes the published numbers.

Everything in ``benchmarks/results.json`` and the README table comes from this module with a fixed seed:
offline ranking metrics for the five models on two datasets under two split protocols, counterfactual
estimator error against exact policy values, the A/B simulator's error rates and the HTTP latency of the
serving path. ``--check`` compares a results file against ``benchmarks/thresholds.json`` and is the CI gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import sklearn
import torch

from curate import __version__
from curate.config import PROJECT_ROOT, Settings
from curate.data.interactions import Dataset
from curate.data.movielens import DatasetUnavailableError, load_movielens
from curate.data.split import Split, split_global_temporal, split_per_user_temporal
from curate.data.synthetic import RetailConfig, RetailWorld, generate_retail
from curate.evaluation.ab import MixtureSPRT, simulate_experiment, slate_hit_probability
from curate.evaluation.counterfactual import (
    BanditFeedback,
    direct_method,
    doubly_robust,
    ips,
    simulate_feedback,
    snips,
    true_policy_value,
)
from curate.evaluation.offline import EvaluationTarget, evaluate_model
from curate.evaluation.policies import SoftmaxPolicy, TabularRewardModel, TopKPolicy
from curate.features.store import FeatureStore
from curate.logging_utils import configure_logging
from curate.pipeline import STORE_FILE, ServingBundle, TrainedModels, fit_for_evaluation, fit_for_serving
from curate.serving.app import create_app
from curate.serving.latency import measure_latency
from curate.serving.state import ServingState

RESULTS_FILE = "results.json"
TABLE_FILE = "results.md"
THRESHOLDS_FILE = "thresholds.json"
MODEL_ORDER = ("popularity", "itemknn", "bpr", "two_tower", "two_stage")
K_VALUES = (10, 20)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "curate": __version__,
        "git_commit": _git_commit(),
    }


def load_datasets(settings: Settings, seed: int) -> tuple[dict[str, Dataset], RetailWorld | None, list[str]]:
    """MovieLens when available plus the synthetic retail set; notes record any fallback."""
    notes: list[str] = []
    datasets: dict[str, Dataset] = {}
    try:
        datasets["movielens_100k"] = load_movielens(settings.data_dir)
    except DatasetUnavailableError as exc:
        notes.append(f"movielens_100k skipped: {exc}")
    retail, world = generate_retail(RetailConfig(seed=seed))
    datasets["retail_synthetic"] = retail
    return datasets, world, notes


def _dataset_summary(dataset: Dataset) -> dict[str, Any]:
    inter = dataset.interactions
    return {
        "n_users": inter.n_users,
        "n_items": inter.n_items,
        "n_interactions": len(inter),
        "density": round(len(inter) / (inter.n_users * inter.n_items), 5),
        "provenance": dataset.provenance,
    }


def evaluate_split(dataset: Dataset, split: Split, seed: int) -> tuple[dict[str, Any], TrainedModels]:
    """Fit every model for one split and score it on the test window."""
    store = FeatureStore()
    try:
        models = fit_for_evaluation(dataset, split, store, seed)
    finally:
        store.close()
    target = EvaluationTarget.from_split(split)
    results = {}
    for name in MODEL_ORDER:
        result = evaluate_model(
            models.all_models()[name], target, ks=K_VALUES, fit_seconds=models.fit_seconds.get(name, 0.0)
        )
        results[name] = result.as_dict()
    return {"split": split.summary(), "n_eval_users": len(target.users), "models": results}, models


def _logging_contexts(rng: np.random.Generator, users: np.ndarray, n_rounds: int) -> np.ndarray:
    return rng.choice(users, size=n_rounds, replace=True)


def counterfactual_block(
    models: TrainedModels, split: Split, expected_reward: np.ndarray, seed: int, n_rounds: int = 50_000
) -> dict[str, Any]:
    """IPS, SNIPS, DM and DR against the exact value of each target policy on simulated logs."""
    rng = np.random.default_rng(seed)
    seen = split.seen_before_test()
    n_items = seen.shape[1]
    target = EvaluationTarget.from_split(split)
    contexts = _logging_contexts(rng, target.users, n_rounds)
    logging_policy = SoftmaxPolicy(
        models.retrievers["popularity"], seen, n_items, temperature=1.0, epsilon=0.2
    )
    feedback = simulate_feedback(rng, logging_policy, expected_reward, contexts)
    reward_model = TabularRewardModel(_fit_reward_table(feedback, expected_reward.shape, seen))
    out: dict[str, Any] = {
        "n_rounds": n_rounds,
        "logging_policy": "softmax(popularity, T=1.0) with 20 percent uniform exploration over unseen items",
        "logging_policy_mean_reward": round(float(feedback.rewards.mean()), 6),
        "policies": {},
    }
    for name in ("popularity", "itemknn", "two_tower", "two_stage"):
        policy = TopKPolicy(models.all_models()[name], seen, n_items, k=10, epsilon=0.05)
        truth = true_policy_value(policy, expected_reward, contexts)
        estimates = {
            "ips": ips(feedback, policy),
            "ips_clipped_50": ips(feedback, policy, clip=50.0),
            "snips": snips(feedback, policy),
            "dm": direct_method(feedback, policy, reward_model),
            "dr": doubly_robust(feedback, policy, reward_model),
        }
        out["policies"][name] = {
            "true_value": round(truth, 6),
            **{
                key: {
                    **est.as_dict(),
                    "abs_error": round(abs(est.value - truth), 6),
                    "covers_truth": bool(est.ci_low <= truth <= est.ci_high),
                }
                for key, est in estimates.items()
            },
        }
    return out


def _fit_reward_table(feedback: BanditFeedback, shape: tuple[int, int], seen: sp.csr_matrix) -> np.ndarray:
    """A deliberately simple reward model: per-item mean logged reward, shrunk towards the global mean."""
    global_mean = float(feedback.rewards.mean())
    sums = np.bincount(feedback.actions, weights=feedback.rewards, minlength=shape[1])
    counts = np.bincount(feedback.actions, minlength=shape[1])
    item_rate = (sums + 5.0 * global_mean) / (counts + 5.0)
    table = np.tile(item_rate, (shape[0], 1))
    coo = seen.tocoo()
    table[coo.row, coo.col] = 0.0
    return table


def _ab_scenario(
    rng: np.random.Generator,
    control: np.ndarray,
    treatment: np.ndarray,
    test: MixtureSPRT,
    label: tuple[str, str],
) -> dict[str, Any]:
    aa = simulate_experiment(
        rng, control, control, n_max=20_000, check_every=100, replications=400, test=test
    )
    ab = simulate_experiment(
        rng, control, treatment, n_max=20_000, check_every=100, replications=400, test=test
    )
    return {
        "control": f"{label[0]} top-10",
        "treatment": f"{label[1]} top-10",
        "control_hit_rate": round(float(control.mean()), 6),
        "treatment_hit_rate": round(float(treatment.mean()), 6),
        "aa": aa.as_dict(),
        "ab": ab.as_dict(),
    }


def ab_block(models: TrainedModels, split: Split, expected_reward: np.ndarray, seed: int) -> dict[str, Any]:
    """A/A and A/B simulations for a large lift (popularity vs two-stage) and a small one (two-tower vs two-stage)."""
    rng = np.random.default_rng(seed)
    seen = split.seen_before_test()
    users = EvaluationTarget.from_split(split).users
    hit = {
        name: slate_hit_probability(models.all_models()[name], users, seen, 10, expected_reward)
        for name in ("popularity", "two_tower", "two_stage")
    }
    test = MixtureSPRT(alpha=0.05, tau=0.02)
    return {
        "alpha": test.alpha,
        "tau": test.tau,
        "replications": 400,
        "n_max": 20_000,
        "check_every": 100,
        "large_lift": _ab_scenario(
            rng, hit["popularity"], hit["two_stage"], test, ("popularity", "two_stage")
        ),
        "small_lift": _ab_scenario(rng, hit["two_tower"], hit["two_stage"], test, ("two_tower", "two_stage")),
    }


def expected_reward_table(
    dataset: Dataset, split: Split, world: RetailWorld | None
) -> tuple[np.ndarray, str]:
    """Ground-truth reward: exact click probabilities for the synthetic world, held-out likes for MovieLens."""
    if world is not None and dataset.name == "retail_synthetic":
        return world.click_matrix(), "exact click probability from the generator's latent utility"
    test = split.test
    liked = test.weights >= 4.0
    table = np.zeros((test.n_users, test.n_items))
    table[test.users[liked], test.items[liked]] = 1.0
    return table, "1 if the user rated the item 4 or more in the test window, else 0"


def latency_block(settings: Settings, dataset: Dataset, seed: int) -> dict[str, Any]:
    """Train the serving bundle for ``dataset`` into the artifact directory and measure HTTP latency."""
    bundle = fit_for_serving(dataset, settings.artifact_dir, seed)
    state = ServingState.from_bundle(
        ServingBundle.load(settings.artifact_dir), FeatureStore(settings.artifact_dir / STORE_FILE), settings
    )
    app = create_app(replace(settings, log_level="WARNING"), state)
    summaries = measure_latency(app, bundle.user_ids, list(MODEL_ORDER), n_requests=200, seed=seed)
    return {
        "dataset": dataset.name,
        "n_requests_per_model": 200,
        "k": 10,
        "explanations": True,
        "transport": "HTTP over loopback to uvicorn in-process",
        "models": {s.model: s.as_dict() for s in summaries},
        "bundle_train_seconds": {k: round(v, 3) for k, v in bundle.train_seconds.items()},
    }


def run_benchmark(settings: Settings, seed: int, output_dir: Path) -> dict[str, Any]:
    """Run every block and write ``results.json`` and ``results.md``."""
    started = time.perf_counter()
    torch.manual_seed(seed)
    datasets, world, notes = load_datasets(settings, seed)
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "environment": _environment(),
        "notes": notes,
        "datasets": {name: _dataset_summary(ds) for name, ds in datasets.items()},
        "offline": {},
        "counterfactual": {},
        "ab_simulator": {},
    }
    for name, dataset in datasets.items():
        per_user = split_per_user_temporal(dataset.interactions)
        offline_per_user, models = evaluate_split(dataset, per_user, seed)
        offline_global, _ = evaluate_split(dataset, split_global_temporal(dataset.interactions), seed)
        results["offline"][name] = {"per_user_temporal": offline_per_user, "global_temporal": offline_global}
        reward, reward_note = expected_reward_table(dataset, per_user, world)
        results["counterfactual"][name] = {
            "reward_model": reward_note,
            **counterfactual_block(models, per_user, reward, seed),
        }
        results["ab_simulator"][name] = {
            "reward_model": reward_note,
            **ab_block(models, per_user, reward, seed),
        }
        print(
            f"[bench] {name}: offline, counterfactual and A/B blocks done ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
    serving_dataset = datasets.get("movielens_100k", datasets["retail_synthetic"])
    results["serving"] = latency_block(settings, serving_dataset, seed)
    results["total_seconds"] = round(time.perf_counter() - started, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RESULTS_FILE).write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / TABLE_FILE).write_text(render_markdown(results))
    print(
        f"[bench] wrote {output_dir / RESULTS_FILE} and {output_dir / TABLE_FILE} in {results['total_seconds']}s"
    )
    return results


def _offline_table(block: dict[str, Any], k: int) -> list[str]:
    lines = [
        f"| model | recall@{k} | ndcg@{k} | hit_rate@{k} | coverage@{k} | novelty@{k} | fit s |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in MODEL_ORDER:
        row = block["models"][name]
        lines.append(
            f"| {name} | {row[f'recall@{k}']:.4f} | {row[f'ndcg@{k}']:.4f} | {row[f'hit_rate@{k}']:.4f} | "
            f"{row[f'coverage@{k}']:.4f} | {row[f'novelty@{k}']:.3f} | {row['fit_seconds']:.1f} |"
        )
    return lines


def render_markdown(results: dict[str, Any]) -> str:
    """Markdown tables for every block; the README copies from this file."""
    lines = [
        f"# Benchmark results (seed {results['seed']}, {results['generated_at']}, commit {results['environment']['git_commit']})",
        "",
    ]
    for name, block in results["offline"].items():
        for protocol, sub in block.items():
            lines += [
                f"## {name}: {protocol} ({sub['n_eval_users']} evaluated users)",
                "",
                *_offline_table(sub, 10),
                "",
            ]
    lines += ["## Counterfactual evaluation (top-10 target policies, 5 percent exploration)", ""]
    for name, block in results["counterfactual"].items():
        lines += [
            f"### {name} ({block['n_rounds']} logged rounds; reward: {block['reward_model']})",
            "",
            "| policy | true value | IPS | SNIPS | DM | DR | IPS ESS |",
            "|---|---|---|---|---|---|---|",
        ]
        for policy, row in block["policies"].items():
            lines.append(
                f"| {policy} | {row['true_value']:.4f} | {row['ips']['value']:.4f} (err {row['ips']['abs_error']:.4f}) | "
                f"{row['snips']['value']:.4f} (err {row['snips']['abs_error']:.4f}) | {row['dm']['value']:.4f} (err {row['dm']['abs_error']:.4f}) | "
                f"{row['dr']['value']:.4f} (err {row['dr']['abs_error']:.4f}) | {row['ips']['effective_sample_size']:.0f} |"
            )
        lines.append("")
    lines += [
        "## A/B simulator (alpha 0.05, 400 replications, up to 20000 arrivals, checks every 100)",
        "",
        "| dataset | scenario | control hit rate | treatment hit rate | A/A mSPRT FPR | A/A naive peeking FPR | "
        "A/A fixed-horizon FPR | A/B mSPRT power | A/B fixed-horizon power | A/B mean n at decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, block in results["ab_simulator"].items():
        for scenario in ("large_lift", "small_lift"):
            sc = block[scenario]
            aa, ab = sc["aa"], sc["ab"]
            lines.append(
                f"| {name} | {sc['control']} vs {sc['treatment']} | {sc['control_hit_rate']:.4f} | "
                f"{sc['treatment_hit_rate']:.4f} | {aa['msprt_reject_rate']:.4f} | {aa['naive_peeking_reject_rate']:.4f} | "
                f"{aa['fixed_horizon_reject_rate']:.4f} | {ab['msprt_treatment_better_rate']:.4f} | "
                f"{ab['fixed_horizon_reject_rate']:.4f} | {ab['msprt_mean_n_at_decision']:.0f} |"
            )
    serving = results["serving"]
    lines += [
        "",
        f"## Serving latency ({serving['dataset']}, {serving['n_requests_per_model']} sequential requests per model, k=10 with explanations)",
        "",
        "| model | p50 ms | p95 ms | p99 ms | mean ms | req/s at concurrency 8 |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in serving["models"].items():
        lines.append(
            f"| {name} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['p99_ms']:.2f} | {row['mean_ms']:.2f} | {row['requests_per_second_c8']:.0f} |"
        )
    lines += ["", f"Total benchmark time: {results['total_seconds']} s.", ""]
    return "\n".join(lines)


README_BEGIN = "<!-- RESULTS:BEGIN -->"
README_END = "<!-- RESULTS:END -->"


def _readme_counterfactual(results: dict[str, Any]) -> list[str]:
    lines = [
        "### Counterfactual estimates of the two-stage top-10 policy",
        "",
        "| dataset | true value | IPS (abs error) | SNIPS (abs error) | DR (abs error) | IPS effective sample size |",
        "|---|---|---|---|---|---|",
    ]
    for name, block in results["counterfactual"].items():
        row = block["policies"]["two_stage"]
        lines.append(
            f"| {name} | {row['true_value']:.4f} | {row['ips']['value']:.4f} ({row['ips']['abs_error']:.4f}) | "
            f"{row['snips']['value']:.4f} ({row['snips']['abs_error']:.4f}) | "
            f"{row['dr']['value']:.4f} ({row['dr']['abs_error']:.4f}) | "
            f"{row['ips']['effective_sample_size']:.0f} of {block['n_rounds']} |"
        )
    return lines


def _readme_ab(results: dict[str, Any]) -> list[str]:
    lines = [
        "### A/B simulator, alpha 0.05, 400 replications of up to 20000 arrivals",
        "",
        "| dataset | control vs treatment | true lift | A/A mSPRT FPR | A/A naive peeking FPR | "
        "A/A fixed-horizon FPR | A/B mSPRT power | mean arrivals at decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, block in results["ab_simulator"].items():
        for scenario in ("large_lift", "small_lift"):
            sc = block[scenario]
            aa, ab = sc["aa"], sc["ab"]
            lines.append(
                f"| {name} | {sc['control']} vs {sc['treatment']} | {ab['true_lift']:+.4f} | "
                f"{aa['msprt_reject_rate']:.4f} | {aa['naive_peeking_reject_rate']:.4f} | "
                f"{aa['fixed_horizon_reject_rate']:.4f} | {ab['msprt_treatment_better_rate']:.4f} | "
                f"{ab['msprt_mean_n_at_decision']:.0f} |"
            )
    return lines


def _readme_serving(results: dict[str, Any]) -> list[str]:
    serving = results["serving"]
    lines = [
        f"### Serving latency ({serving['dataset']}, {serving['n_requests_per_model']} sequential HTTP requests "
        "per model, k = 10, explanations on)",
        "",
        "| model | p50 ms | p95 ms | p99 ms | req/s at concurrency 8 |",
        "|---|---|---|---|---|",
    ]
    for name, row in serving["models"].items():
        lines.append(
            f"| {name} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | {row['p99_ms']:.2f} | "
            f"{row['requests_per_second_c8']:.0f} |"
        )
    return lines


def render_readme_section(results: dict[str, Any]) -> str:
    """The results section of the README, generated so the numbers cannot drift from ``results.json``."""
    env = results["environment"]
    lines = [
        f"Seed {results['seed']}, generated {results['generated_at']}, commit `{env['git_commit']}`, "
        f"Python {env['python']}, torch {env['torch']}, scikit-learn {env['scikit_learn']} on {env['machine']}. "
        f"Total benchmark time {results['total_seconds']} s.",
        "",
    ]
    for name, block in results["offline"].items():
        sub = block["per_user_temporal"]
        lines += [
            f"### {name}: per-user temporal split, k = 10 ({sub['n_eval_users']} evaluated users)",
            "",
            *_offline_table(sub, 10),
            "",
        ]
    lines += [*_readme_counterfactual(results), "", *_readme_ab(results), "", *_readme_serving(results)]
    lines += [
        "",
        "Full tables including k = 20, the global temporal split and every estimator: "
        "[benchmarks/results.md](benchmarks/results.md).",
    ]
    return "\n".join(lines)


def update_readme(results: dict[str, Any], readme: Path) -> None:
    """Replace the block between the README markers with the rendered results section."""
    text = readme.read_text()
    start, end = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
    readme.write_text(text[:start] + "\n" + render_readme_section(results) + "\n" + text[end:])


def _lookup(results: dict[str, Any], path: str) -> float:
    node: Any = results
    for part in path.split("."):
        node = node[part]
    return float(node)


def check_thresholds(output_dir: Path) -> bool:
    """Compare ``results.json`` against ``thresholds.json``; print one line per check and return the verdict."""
    results = json.loads((output_dir / RESULTS_FILE).read_text())
    thresholds = json.loads((output_dir / THRESHOLDS_FILE).read_text())
    passed = True
    for check in thresholds["checks"]:
        try:
            value = _lookup(results, check["path"])
        except KeyError:
            if check.get("optional"):
                print(f"SKIP {check['path']} (not in results)")
                continue
            print(f"FAIL {check['path']} missing from results")
            passed = False
            continue
        ok = value >= check["min"] if "min" in check else value <= check["max"]
        bound = f">= {check['min']}" if "min" in check else f"<= {check['max']}"
        print(f"{'PASS' if ok else 'FAIL'} {check['path']} = {value:.4f} (want {bound})")
        passed &= ok
    print("quality gate:", "PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run the benchmark, or ``--check`` an existing results file."""
    parser = argparse.ArgumentParser(description="curate benchmark harness")
    parser.add_argument("--seed", type=int, default=Settings.from_env().seed)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare results.json against thresholds.json and exit non-zero on failure",
    )
    parser.add_argument(
        "--update-readme", action="store_true", help="copy the measured tables into README.md"
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    configure_logging("WARNING")
    if args.check:
        return 0 if check_thresholds(args.output) else 1
    if args.update_readme:
        update_readme(json.loads((args.output / RESULTS_FILE).read_text()), PROJECT_ROOT / "README.md")
        print("README results section updated")
        return 0
    run_benchmark(settings, args.seed, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
