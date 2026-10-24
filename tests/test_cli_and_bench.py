import json
from pathlib import Path

import pytest

from curate.bench import check_thresholds, render_markdown
from curate.cli import build_parser, main
from curate.config import Settings


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CURATE_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("APP_API_KEY", "k")
    monkeypatch.setenv("CURATE_SEED", "7")
    monkeypatch.setenv("CURATE_LLM_EXPLAIN", "true")
    settings = Settings.from_env()
    assert settings.data_dir == tmp_path / "d"
    assert settings.api_key == "k"
    assert settings.seed == 7
    assert settings.llm_explanations is True
    monkeypatch.setenv("APP_API_KEY", "")
    assert Settings.from_env().api_key is None


def test_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["recommend", "--user", "3", "--k", "4"])
    assert args.command == "recommend"
    assert (args.user, args.k) == (3, 4)
    assert parser.parse_args(["bench"]).output == "benchmarks"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--dataset", "unknown"])


def test_cli_data_offline_uses_synthetic_and_reports_missing_movielens(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CURATE_DATA_DIR", str(tmp_path))
    assert main(["data", "--offline", "--seed", "3"]) == 0
    captured = capsys.readouterr()
    assert "retail_synthetic" in captured.out
    assert "unavailable" in captured.err


def _fake_results():
    model_row = {
        "recall@10": 0.2,
        "ndcg@10": 0.2,
        "hit_rate@10": 0.5,
        "coverage@10": 0.3,
        "novelty@10": 3.0,
        "fit_seconds": 1.0,
    }
    sub = {
        "n_eval_users": 10,
        "split": {},
        "models": {m: dict(model_row) for m in ("popularity", "itemknn", "bpr", "two_tower", "two_stage")},
    }
    est = {"value": 0.1, "abs_error": 0.01, "effective_sample_size": 100.0}
    policy = {"true_value": 0.1, "ips": est, "snips": est, "dm": est, "dr": est}
    ab = {
        "msprt_reject_rate": 0.02,
        "naive_peeking_reject_rate": 0.4,
        "fixed_horizon_reject_rate": 0.05,
        "msprt_treatment_better_rate": 0.9,
        "msprt_mean_n_at_decision": 5000.0,
    }
    return {
        "seed": 1,
        "generated_at": "now",
        "environment": {"git_commit": "abc"},
        "offline": {"retail_synthetic": {"per_user_temporal": sub, "global_temporal": sub}},
        "counterfactual": {
            "retail_synthetic": {"n_rounds": 10, "reward_model": "x", "policies": {"two_stage": policy}}
        },
        "ab_simulator": {
            "retail_synthetic": {"control_hit_rate": 0.1, "treatment_hit_rate": 0.2, "aa": ab, "ab": ab}
        },
        "serving": {
            "dataset": "retail_synthetic",
            "n_requests_per_model": 5,
            "models": {
                "two_stage": {
                    "p50_ms": 1.0,
                    "p95_ms": 2.0,
                    "p99_ms": 3.0,
                    "mean_ms": 1.5,
                    "requests_per_second_c8": 100.0,
                }
            },
        },
        "total_seconds": 1.0,
    }


def test_render_markdown_contains_every_block():
    text = render_markdown(_fake_results())
    assert "| two_stage | 0.2000 |" in text
    assert "## Counterfactual evaluation" in text
    assert "## A/B simulator" in text
    assert "## Serving latency" in text


def test_check_thresholds_pass_fail_and_optional(tmp_path: Path, capsys):
    (tmp_path / "results.json").write_text(json.dumps(_fake_results()))
    thresholds = {
        "checks": [
            {"path": "offline.retail_synthetic.per_user_temporal.models.two_stage.recall@10", "min": 0.1},
            {"path": "ab_simulator.retail_synthetic.aa.msprt_reject_rate", "max": 0.08},
            {
                "path": "offline.movielens_100k.per_user_temporal.models.two_stage.recall@10",
                "min": 0.1,
                "optional": True,
            },
        ]
    }
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    assert check_thresholds(tmp_path) is True
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "SKIP" in out
    thresholds["checks"][0]["min"] = 0.9
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    assert check_thresholds(tmp_path) is False
    thresholds["checks"] = [{"path": "offline.missing.x", "min": 0}]
    (tmp_path / "thresholds.json").write_text(json.dumps(thresholds))
    assert check_thresholds(tmp_path) is False


def test_repository_thresholds_file_is_well_formed():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "thresholds.json"
    checks = json.loads(path.read_text())["checks"]
    assert len(checks) >= 8
    for check in checks:
        assert "path" in check
        assert "min" in check or "max" in check
