"""Command line interface: ``curate data | train | serve | recommend | bench | check``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from curate.config import Settings
from curate.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def cmd_data(args: argparse.Namespace, settings: Settings) -> int:
    """Download and verify MovieLens 100k and generate the synthetic dataset once."""
    from curate.data.movielens import DatasetUnavailableError, load_movielens
    from curate.data.synthetic import RetailConfig, generate_retail

    try:
        movielens = load_movielens(settings.data_dir, download=not args.offline)
        print(f"movielens_100k: {len(movielens.interactions)} interactions, sha256 verified")
    except DatasetUnavailableError as exc:
        print(f"movielens_100k unavailable: {exc}", file=sys.stderr)
    retail, _ = generate_retail(RetailConfig(seed=args.seed))
    print(f"retail_synthetic: {len(retail.interactions)} interactions (seed {args.seed})")
    return 0


def _load_dataset(name: str, settings: Settings, seed: int):  # noqa: ANN202
    from curate.data.movielens import load_movielens
    from curate.data.synthetic import RetailConfig, generate_retail

    if name == "movielens_100k":
        return load_movielens(settings.data_dir)
    return generate_retail(RetailConfig(seed=seed))[0]


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    """Fit the serving bundle into the artifact directory."""
    from curate.pipeline import fit_for_serving

    dataset = _load_dataset(args.dataset, settings, args.seed)
    bundle = fit_for_serving(dataset, settings.artifact_dir, args.seed)
    print(
        json.dumps(
            {
                "dataset": bundle.dataset_name,
                "models": sorted(bundle.models),
                "train_seconds": bundle.train_seconds,
            },
            indent=2,
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """Serve the API, training a bundle first when none exists."""
    import uvicorn

    from curate.pipeline import BUNDLE_FILE
    from curate.serving.app import create_app

    if not (settings.artifact_dir / BUNDLE_FILE).exists():
        logger.info("no bundle found, training one first")
        cmd_train(args, settings)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)
    return 0


def cmd_recommend(args: argparse.Namespace, settings: Settings) -> int:
    """Print recommendations for one user from the saved bundle."""
    from curate.serving.state import ServingState

    state = ServingState.load(settings)
    result = state.recommend(args.user, args.k, args.model, explain=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_bench(args: argparse.Namespace, settings: Settings) -> int:
    """Run the benchmark harness."""
    from curate.bench import run_benchmark

    run_benchmark(settings, seed=args.seed, output_dir=Path(args.output))
    return 0


def cmd_check(args: argparse.Namespace, settings: Settings) -> int:  # noqa: ARG001
    """Compare results against thresholds."""
    from curate.bench import check_thresholds

    return 0 if check_thresholds(Path(args.output)) else 1


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="curate", description="Recommender platform with offline and counterfactual evaluation"
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=Settings.from_env().seed)
    common.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="download and verify datasets", parents=[common])
    data.add_argument(
        "--offline", action="store_true", help="do not download; fail if MovieLens is not cached"
    )
    data.set_defaults(func=cmd_data)
    for name, func in (("train", cmd_train), ("serve", cmd_serve)):
        p = sub.add_parser(name, help=func.__doc__, parents=[common])
        p.add_argument("--dataset", choices=("movielens_100k", "retail_synthetic"), default="movielens_100k")
        p.set_defaults(func=func)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
    rec = sub.add_parser("recommend", help="recommend for one user from the saved bundle", parents=[common])
    rec.add_argument("--user", type=int, required=True)
    rec.add_argument("--k", type=int, default=10)
    rec.add_argument("--model", default="two_stage")
    rec.set_defaults(func=cmd_recommend)
    for name, func in (("bench", cmd_bench), ("check", cmd_check)):
        p = sub.add_parser(name, help=func.__doc__, parents=[common])
        p.add_argument("--output", default="benchmarks")
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    configure_logging(args.log_level or settings.log_level)
    return int(args.func(args, settings))


if __name__ == "__main__":
    sys.exit(main())
