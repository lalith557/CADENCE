"""Yelp review loader — real vocabulary drift over time.

Reads yelp_academic_dataset_review.json line-by-line so we never hold
the full 6.9 M-line file in memory. Returns two disjoint time slices
(early vs late by review `date`) plus the target label — binary
sentiment (positive if stars >= 4, negative if stars <= 2, neutral 3
dropped).

Genuine drift signal: Yelp vocabulary shifts noticeably across years
(new emoji patterns, new business types, changing service-industry
language), so a classifier trained on 2013 reviews degrades
measurably on 2019 reviews without any injected drift.

The loader caps the row count per slice so a Phase-F text run finishes
in minutes rather than hours; the whole dataset is available if you
want a paper-grade run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class YelpSlice:
    name: str
    texts: list[str]
    y: np.ndarray  # int64 {0, 1}
    dates: list[str]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n": len(self.texts),
            "positive_rate": float(np.mean(self.y)) if self.y.size else 0.0,
            "date_min": min(self.dates) if self.dates else None,
            "date_max": max(self.dates) if self.dates else None,
        }


DEFAULT_PATH = Path(
    "Dataset/Yelp JSON Dataset1/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json"
)


def _iter_reviews(path: Path, *, max_lines: int) -> Iterator[dict]:
    """Stream up to `max_lines` reviews from the raw JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                return
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_yelp_slices(
    path: Path | str = DEFAULT_PATH,
    *,
    early_year_max: int = 2013,
    late_year_min: int = 2018,
    per_slice_cap: int = 20_000,
    max_lines_scan: int = 500_000,
    seed: int = 42,
) -> tuple[YelpSlice, YelpSlice]:
    """Return (early_slice, late_slice).

    The scan bails out after `max_lines_scan` raw lines OR once both
    slices are full. On a laptop this reads ~500 k lines in ~30 s and
    produces two 20 k-row slices with genuinely disjoint years.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Yelp review JSON not found at {path}. See docs/README/CADENCE-datasets-verified.md."
        )

    rng = np.random.default_rng(seed)
    early: list[dict] = []
    late: list[dict] = []
    scanned = 0
    for r in _iter_reviews(path, max_lines=max_lines_scan):
        scanned += 1
        stars = r.get("stars")
        text = r.get("text")
        date = r.get("date")
        if not isinstance(text, str) or not text or stars is None or not isinstance(date, str):
            continue
        try:
            year = int(date[:4])
        except ValueError:
            continue
        # Binary sentiment: drop neutral 3.
        if stars <= 2:
            y = 0
        elif stars >= 4:
            y = 1
        else:
            continue
        if year <= early_year_max and len(early) < per_slice_cap:
            early.append({"text": text, "y": y, "date": date})
        elif year >= late_year_min and len(late) < per_slice_cap:
            late.append({"text": text, "y": y, "date": date})
        if len(early) >= per_slice_cap and len(late) >= per_slice_cap:
            break

    def _shuffle(rows: list[dict], name: str) -> YelpSlice:
        idx = rng.permutation(len(rows))
        rows = [rows[i] for i in idx]
        return YelpSlice(
            name=name,
            texts=[r["text"] for r in rows],
            y=np.array([r["y"] for r in rows], dtype=np.int64),
            dates=[r["date"] for r in rows],
        )

    return _shuffle(early, f"yelp_early_{early_year_max}"), _shuffle(late, f"yelp_late_{late_year_min}")


def load_yelp_domain_shift(
    path: Path | str = DEFAULT_PATH,
    *,
    per_slice_cap: int = 10_000,
    max_lines_scan: int = 500_000,
    seed: int = 42,
) -> tuple[YelpSlice, YelpSlice]:
    """Deliberate domain shift — train on EXTREME reviews (stars ∈ {1, 5}),
    stream on MARGINAL reviews (stars ∈ {2, 4}). Same binary sentiment
    (2 → 0, 4 → 1, so we relabel here) but the model trained on 1★/5★
    reviews has seen only vocabulary and phrasing from extreme opinions.

    On 2★/4★ reviews the reviewer is much more measured ("okay but",
    "decent, though") — vocabulary the classifier under-weighted. This
    produces a measurable F1 drop on a shallow classifier, whereas the
    naive year-based split we tried first did not.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Yelp review JSON not found at {path}.")

    rng = np.random.default_rng(seed)
    extreme: list[dict] = []
    marginal: list[dict] = []
    for r in _iter_reviews(path, max_lines=max_lines_scan):
        stars = r.get("stars")
        text = r.get("text")
        date = r.get("date")
        if not isinstance(text, str) or not text or stars is None or not isinstance(date, str):
            continue
        if stars in (1, 1.0) and len(extreme) < per_slice_cap:
            extreme.append({"text": text, "y": 0, "date": date})
        elif stars in (5, 5.0) and len(extreme) < per_slice_cap:
            extreme.append({"text": text, "y": 1, "date": date})
        elif stars in (2, 2.0) and len(marginal) < per_slice_cap:
            marginal.append({"text": text, "y": 0, "date": date})
        elif stars in (4, 4.0) and len(marginal) < per_slice_cap:
            marginal.append({"text": text, "y": 1, "date": date})
        if len(extreme) >= per_slice_cap and len(marginal) >= per_slice_cap:
            break

    def _shuffle(rows: list[dict], name: str) -> YelpSlice:
        idx = rng.permutation(len(rows))
        rows = [rows[i] for i in idx]
        return YelpSlice(
            name=name,
            texts=[r["text"] for r in rows],
            y=np.array([r["y"] for r in rows], dtype=np.int64),
            dates=[r["date"] for r in rows],
        )

    return _shuffle(extreme, "yelp_extreme_1_5"), _shuffle(marginal, "yelp_marginal_2_4")
