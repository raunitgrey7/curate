"""MovieLens 100k: checksum-verified download, cached extraction and parsing into a ``Dataset``.

The archive is fetched once into ``data/ml-100k.zip``, verified against a SHA-256 recorded in this module
and in ``data/CHECKSUMS.sha256``, then only the files we read are extracted. Ratings are kept as weights
but every rating counts as a positive "watched" event; the evaluation protocol documents this choice.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from curate.data.interactions import Catalog, Dataset, Interactions, UserTable

logger = logging.getLogger(__name__)

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
ML100K_MD5_PUBLISHED = "0e33842e24a9c977be4e0107933c0723"
ARCHIVE_NAME = "ml-100k.zip"
NEEDED_FILES = ("u.data", "u.item", "u.user", "u.genre", "u.occupation", "README")
AGE_BUCKETS = (18, 25, 35, 45, 50, 56)
GENRE_COUNT = 19


class ChecksumError(RuntimeError):
    """Raised when a downloaded archive does not match the recorded digest."""


class DatasetUnavailableError(RuntimeError):
    """Raised when the dataset is neither cached nor downloadable."""


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str = ML100K_SHA256) -> str:
    """Return the digest of ``path`` or raise ``ChecksumError`` when it differs from ``expected``."""
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(f"{path.name}: sha256 {actual} does not match recorded {expected}")
    return actual


def download_archive(data_dir: Path, url: str = ML100K_URL, timeout: float = 60.0) -> Path:
    """Download the zip to a temporary file, verify it, then move it into ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / ARCHIVE_NAME
    with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        logger.info("downloading MovieLens 100k", extra={"url": url})
        with urllib.request.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        verify_checksum(tmp_path)
        tmp_path.replace(target)
        target.chmod(0o644)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def extract_archive(archive: Path, data_dir: Path) -> Path:
    """Extract the needed members into ``data_dir/ml-100k`` while rejecting unsafe member names."""
    out_dir = data_dir / "ml-100k"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or name not in NEEDED_FILES or ".." in member.filename:
                continue
            with zf.open(member) as src, (out_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    missing = [name for name in NEEDED_FILES if not (out_dir / name).exists()]
    if missing:
        raise DatasetUnavailableError(f"archive is missing {missing}")
    return out_dir


def ensure_movielens(data_dir: Path, download: bool = True) -> Path:
    """Return the extracted ``ml-100k`` directory, downloading and verifying the archive when needed."""
    out_dir = data_dir / "ml-100k"
    if all((out_dir / name).exists() for name in NEEDED_FILES):
        return out_dir
    archive = data_dir / ARCHIVE_NAME
    if archive.exists():
        try:
            verify_checksum(archive)
        except ChecksumError:
            logger.warning("cached archive failed verification, deleting it")
            archive.unlink()
    if not archive.exists():
        if not download:
            raise DatasetUnavailableError("MovieLens archive is not cached and download is disabled")
        try:
            download_archive(data_dir)
        except ChecksumError:
            raise
        except OSError as exc:
            raise DatasetUnavailableError(f"download failed: {exc}") from exc
    return extract_archive(archive, data_dir)


def _parse_items(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    ids: list[int] = []
    titles: list[str] = []
    years: list[int] = []
    flags: list[list[int]] = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 5 + GENRE_COUNT:
                continue
            ids.append(int(parts[0]))
            titles.append(parts[1])
            date = parts[2]
            years.append(int(date[-4:]) if len(date) >= 4 and date[-4:].isdigit() else 0)
            flags.append([int(flag) for flag in parts[5 : 5 + GENRE_COUNT]])
    return np.asarray(ids), titles, np.asarray(flags, dtype=np.float32), np.asarray(years, dtype=np.int64)


def _parse_genres(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if "|" in line:
                names.append(line.split("|")[0].strip())
    return names


def _parse_users(path: Path, occupations: list[str]) -> UserTable:
    occupation_index = {name: i for i, name in enumerate(occupations)}
    age_names = [f"age_lt_{AGE_BUCKETS[0]}"] + [
        f"age_{lo}_{hi - 1}" for lo, hi in itertools.pairwise(AGE_BUCKETS)
    ]
    age_names.append(f"age_ge_{AGE_BUCKETS[-1]}")
    names = age_names + ["gender_f", "gender_m"] + [f"occ_{name}" for name in occupations]
    ids: list[int] = []
    rows: list[np.ndarray] = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            uid, age, gender, occupation, _zip = line.rstrip("\n").split("|")
            row = np.zeros(len(names), dtype=np.float32)
            row[int(np.searchsorted(AGE_BUCKETS, int(age), side="right"))] = 1.0
            row[len(age_names) + (0 if gender == "F" else 1)] = 1.0
            row[len(age_names) + 2 + occupation_index[occupation]] = 1.0
            ids.append(int(uid))
            rows.append(row)
    return UserTable(user_ids=np.asarray(ids), features=np.vstack(rows), feature_names=names)


def load_movielens(data_dir: Path, download: bool = True) -> Dataset:
    """Load MovieLens 100k as a ``Dataset`` with contiguous indices and side tables."""
    root = ensure_movielens(data_dir, download=download)
    item_ids, titles, genres, years = _parse_items(root / "u.item")
    genre_names = _parse_genres(root / "u.genre")
    occupations = [
        line.strip()
        for line in (root / "u.occupation").read_text(encoding="latin-1").splitlines()
        if line.strip()
    ]
    users = _parse_users(root / "u.user", occupations)
    raw = np.array((root / "u.data").read_text(encoding="latin-1").split(), dtype=np.int64).reshape(-1, 4)
    user_pos = {int(u): i for i, u in enumerate(users.user_ids)}
    item_pos = {int(i): j for j, i in enumerate(item_ids)}
    interactions = Interactions(
        users=np.fromiter((user_pos[int(u)] for u in raw[:, 0]), dtype=np.int64, count=len(raw)),
        items=np.fromiter((item_pos[int(i)] for i in raw[:, 1]), dtype=np.int64, count=len(raw)),
        timestamps=raw[:, 3],
        weights=raw[:, 2].astype(np.float32),
        n_users=len(users.user_ids),
        n_items=len(item_ids),
    )
    catalog = Catalog(item_ids=item_ids, titles=titles, genres=genres, genre_names=genre_names, years=years)
    provenance = {"source": ML100K_URL, "sha256": ML100K_SHA256, "md5_published": ML100K_MD5_PUBLISHED}
    return Dataset(
        name="movielens_100k", interactions=interactions, catalog=catalog, users=users, provenance=provenance
    )
