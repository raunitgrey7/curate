import hashlib
import zipfile
from pathlib import Path

import pytest

from curate.config import PROJECT_ROOT
from curate.data.movielens import (
    ML100K_SHA256,
    ChecksumError,
    DatasetUnavailableError,
    ensure_movielens,
    extract_archive,
    load_movielens,
    verify_checksum,
)

CACHED = (PROJECT_ROOT / "data" / "ml-100k" / "u.data").exists()


def _fake_archive(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    (root / "u.data").write_text("1\t1\t5\t100\n1\t2\t3\t200\n2\t1\t4\t150\n")
    (root / "u.item").write_text(
        "1|Toy Story (1995)|01-Jan-1995||url|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        "2|Caf\xe9 (1998)|02-Feb-1998||url|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n",
        encoding="latin-1",
    )
    (root / "u.user").write_text("1|24|M|technician|85711\n2|53|F|other|94043\n")
    (root / "u.genre").write_text("unknown|0\nAction|1\n" + "".join(f"g{i}|{i}\n" for i in range(2, 19)))
    (root / "u.occupation").write_text("technician\nother\n")
    (root / "README").write_text("fake")
    archive = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("u.data", "u.item", "u.user", "u.genre", "u.occupation", "README"):
            zf.write(root / name, f"ml-100k/{name}")
        zf.writestr("ml-100k/../evil.txt", "nope")
    return archive


def test_verify_checksum_rejects_tampered_file(tmp_path):
    path = tmp_path / "x.zip"
    path.write_bytes(b"not the dataset")
    with pytest.raises(ChecksumError, match="does not match"):
        verify_checksum(path)
    assert verify_checksum(path, hashlib.sha256(b"not the dataset").hexdigest())


def test_extract_skips_unsafe_members_and_parses(tmp_path):
    archive = _fake_archive(tmp_path)
    data_dir = tmp_path / "data"
    out = extract_archive(archive, data_dir)
    assert not (data_dir / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()
    dataset = load_movielens(data_dir, download=False)
    assert out == data_dir / "ml-100k"
    assert dataset.interactions.n_users == 2
    assert dataset.interactions.n_items == 2
    assert dataset.catalog.titles[1] == "Caf\xe9 (1998)"
    assert dataset.catalog.years.tolist() == [1995, 1998]
    assert dataset.catalog.genre_labels(0) == ["g3", "g4", "g5"]
    assert dataset.users.features.shape == (2, 7 + 2 + 2)
    assert dataset.users.features[0].sum() == 3.0
    assert dataset.interactions.weights.tolist() == [5.0, 3.0, 4.0]


def test_offline_without_cache_raises(tmp_path):
    with pytest.raises(DatasetUnavailableError):
        ensure_movielens(tmp_path, download=False)


def test_corrupt_cached_archive_is_removed_and_reported(tmp_path):
    (tmp_path / "ml-100k.zip").write_bytes(b"corrupt")
    with pytest.raises(DatasetUnavailableError):
        ensure_movielens(tmp_path, download=False)
    assert not (tmp_path / "ml-100k.zip").exists()


@pytest.mark.skipif(not CACHED, reason="MovieLens is not cached under data/")
def test_cached_movielens_matches_published_shape():
    dataset = load_movielens(PROJECT_ROOT / "data", download=False)
    assert len(dataset.interactions) == 100_000
    assert dataset.interactions.n_users == 943
    assert dataset.interactions.n_items == 1682
    assert dataset.provenance["sha256"] == ML100K_SHA256
    assert dataset.interactions.weights.min() >= 1
    assert dataset.interactions.weights.max() <= 5


@pytest.mark.skipif(not (PROJECT_ROOT / "data" / "ml-100k.zip").exists(), reason="archive not cached")
def test_cached_archive_has_recorded_checksum():
    assert verify_checksum(PROJECT_ROOT / "data" / "ml-100k.zip") == ML100K_SHA256
    recorded = (PROJECT_ROOT / "data" / "CHECKSUMS.sha256").read_text().split()[0]
    assert recorded == ML100K_SHA256
