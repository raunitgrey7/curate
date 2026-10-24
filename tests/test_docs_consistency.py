"""The README table must be the benchmark output, not a hand-written copy."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results.json"
README = ROOT / "README.md"
BANNED = (chr(0x2014), chr(0x2013))  # em dash, en dash


def _results():
    if not RESULTS.exists():
        pytest.skip("benchmarks/results.json has not been generated")
    return json.loads(RESULTS.read_text())


def test_readme_offline_table_matches_results_json():
    results = _results()
    text = README.read_text()
    block = results["offline"]["movielens_100k"]["per_user_temporal"]["models"]
    for model, row in block.items():
        pattern = (
            rf"^\| {re.escape(model)} \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|"
        )
        match = re.search(pattern, text, flags=re.MULTILINE)
        assert match, f"README has no MovieLens row for {model}"
        recall, ndcg, hit_rate, coverage, novelty = (float(x) for x in match.groups())
        assert recall == pytest.approx(row["recall@10"], abs=5e-5)
        assert ndcg == pytest.approx(row["ndcg@10"], abs=5e-5)
        assert hit_rate == pytest.approx(row["hit_rate@10"], abs=5e-5)
        assert coverage == pytest.approx(row["coverage@10"], abs=5e-5)
        assert novelty == pytest.approx(row["novelty@10"], abs=5e-4)


def test_readme_latency_matches_results_json():
    results = _results()
    text = README.read_text()
    section = text[text.index("### Serving latency") :]
    row = results["serving"]["models"]["two_stage"]
    match = re.search(r"^\| two_stage \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|", section, flags=re.MULTILINE)
    assert match, "README has no latency row for two_stage"
    p50, p95, p99 = (float(x) for x in match.groups())
    assert (p50, p95, p99) == (round(row["p50_ms"], 2), round(row["p95_ms"], 2), round(row["p99_ms"], 2))


@pytest.mark.parametrize("path", ["README.md", "REPORT.md", "docs/ARCHITECTURE.md", "docs/EVALUATION.md"])
def test_prose_has_no_dashes_of_the_banned_kind(path):
    file = ROOT / path
    if not file.exists():
        pytest.skip(f"{path} not present")
    text = file.read_text()
    for char in BANNED:
        assert char not in text, f"{path} contains U+{ord(char):04X}"


def test_source_has_no_banned_dashes():
    offenders = []
    for file in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        text = file.read_text()
        if any(ch in text for ch in BANNED):
            offenders.append(str(file.relative_to(ROOT)))
    assert offenders == []
