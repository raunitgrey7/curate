"""A small sqlite feature store.

Feature vectors are stored as float32 blobs keyed by (feature set, entity id), with the feature names kept
once per set, so reading a whole set back as a matrix is one query and one ``frombuffer``. Entity metadata
(titles, genres) is JSON per entity. sqlite gives us durability, a single-file artifact that ships inside
the Docker image and point lookups fast enough for the request path.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity TEXT NOT NULL,
    name TEXT NOT NULL,
    feature_names TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE (entity, name)
);
CREATE TABLE IF NOT EXISTS feature_values (
    feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (feature_set_id, entity_id)
);
CREATE TABLE IF NOT EXISTS entity_metadata (
    entity TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (entity, entity_id)
);
"""


@dataclass(frozen=True)
class FeatureSet:
    """Descriptor of one stored feature set."""

    id: int
    entity: str
    name: str
    feature_names: list[str]
    created_at: float


class FeatureStore:
    """Read and write feature vectors and metadata for users and items."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def write_features(
        self, entity: str, name: str, ids: np.ndarray, names: list[str], matrix: np.ndarray
    ) -> FeatureSet:
        """Replace the feature set ``(entity, name)`` with the given vectors."""
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.shape != (len(ids), len(names)):
            raise ValueError("matrix shape must be (len(ids), len(names))")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("feature matrix contains non-finite values")
        with self._conn:
            self._conn.execute("DELETE FROM feature_sets WHERE entity = ? AND name = ?", (entity, name))
            cursor = self._conn.execute(
                "INSERT INTO feature_sets (entity, name, feature_names, created_at) VALUES (?, ?, ?, ?)",
                (entity, name, json.dumps(names), time.time()),
            )
            set_id = int(cursor.lastrowid or 0)
            self._conn.executemany(
                "INSERT INTO feature_values (feature_set_id, entity_id, vector) VALUES (?, ?, ?)",
                ((set_id, int(i), row.tobytes()) for i, row in zip(ids, matrix, strict=True)),
            )
        return self._feature_set(entity, name)

    def _feature_set(self, entity: str, name: str) -> FeatureSet:
        row = self._conn.execute(
            "SELECT id, entity, name, feature_names, created_at FROM feature_sets WHERE entity = ? AND name = ?",
            (entity, name),
        ).fetchone()
        if row is None:
            raise KeyError(f"feature set {entity}/{name} does not exist")
        return FeatureSet(int(row[0]), row[1], row[2], json.loads(row[3]), float(row[4]))

    def list_feature_sets(self) -> list[FeatureSet]:
        """All stored feature sets, oldest first."""
        rows = self._conn.execute(
            "SELECT id, entity, name, feature_names, created_at FROM feature_sets ORDER BY id"
        ).fetchall()
        return [FeatureSet(int(r[0]), r[1], r[2], json.loads(r[3]), float(r[4])) for r in rows]

    def read_features(self, entity: str, name: str) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return ``(ids, feature_names, matrix)`` for a whole feature set, ordered by entity id."""
        feature_set = self._feature_set(entity, name)
        rows = self._conn.execute(
            "SELECT entity_id, vector FROM feature_values WHERE feature_set_id = ? ORDER BY entity_id",
            (feature_set.id,),
        ).fetchall()
        ids = np.asarray([r[0] for r in rows], dtype=np.int64)
        width = len(feature_set.feature_names)
        matrix = (
            np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            if rows
            else np.empty((0, width), dtype=np.float32)
        )
        return ids, feature_set.feature_names, matrix

    def get_vector(self, entity: str, name: str, entity_id: int) -> dict[str, float] | None:
        """One entity's features as a name to value mapping, or ``None`` when absent."""
        feature_set = self._feature_set(entity, name)
        row = self._conn.execute(
            "SELECT vector FROM feature_values WHERE feature_set_id = ? AND entity_id = ?",
            (feature_set.id, int(entity_id)),
        ).fetchone()
        if row is None:
            return None
        values = np.frombuffer(row[0], dtype=np.float32)
        return {n: float(v) for n, v in zip(feature_set.feature_names, values, strict=True)}

    def write_metadata(self, entity: str, ids: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        """Upsert JSON metadata for many entities."""
        if len(ids) != len(payloads):
            raise ValueError("ids and payloads must have the same length")
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO entity_metadata (entity, entity_id, payload) VALUES (?, ?, ?)",
                (
                    (entity, int(i), json.dumps(p, ensure_ascii=False))
                    for i, p in zip(ids, payloads, strict=True)
                ),
            )

    def get_metadata(self, entity: str, entity_id: int) -> dict[str, Any] | None:
        """Metadata for one entity, or ``None`` when absent."""
        row = self._conn.execute(
            "SELECT payload FROM entity_metadata WHERE entity = ? AND entity_id = ?", (entity, int(entity_id))
        ).fetchone()
        return json.loads(row[0]) if row else None

    def count(self, entity: str) -> int:
        """Number of entities with metadata."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM entity_metadata WHERE entity = ?", (entity,)
        ).fetchone()
        return int(row[0])
