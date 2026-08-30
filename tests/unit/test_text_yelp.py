"""Guards for cadence.data.text_yelp — loader path + domain-shift split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadence.data.text_yelp import (
    DEFAULT_PATH,
    YelpSlice,
    load_yelp_domain_shift,
    load_yelp_slices,
)


def _yelp_available() -> bool:
    return DEFAULT_PATH.exists()


@pytest.mark.needs_dataset
@pytest.mark.skipif(not _yelp_available(), reason="Yelp JSON not on disk")
def test_load_yelp_slices_produces_disjoint_year_ranges() -> None:
    early, late = load_yelp_slices(per_slice_cap=1000, max_lines_scan=20_000)
    assert isinstance(early, YelpSlice)
    assert isinstance(late, YelpSlice)
    assert early.y.shape[0] > 0
    assert late.y.shape[0] > 0
    # Early slice year <= 2013, late slice year >= 2018.
    early_years = {int(d[:4]) for d in early.dates}
    late_years = {int(d[:4]) for d in late.dates}
    assert max(early_years) <= 2013
    assert min(late_years) >= 2018


@pytest.mark.needs_dataset
@pytest.mark.skipif(not _yelp_available(), reason="Yelp JSON not on disk")
def test_load_yelp_domain_shift_isolates_star_ratings() -> None:
    extreme, marginal = load_yelp_domain_shift(per_slice_cap=800, max_lines_scan=20_000)
    # Extreme slice has both classes (1★ → 0 and 5★ → 1); marginal likewise
    # from 2★/4★. Cannot check star directly (dropped) but positive rates
    # should be non-trivial for both.
    assert 0.05 < float(extreme.y.mean()) < 0.95
    assert 0.05 < float(marginal.y.mean()) < 0.95


def test_load_yelp_slices_raises_on_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Yelp"):
        load_yelp_slices(path=tmp_path / "no-such-file.json", per_slice_cap=100, max_lines_scan=100)
