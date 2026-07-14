"""Per-track evaluation regimes — the pinned ``(fill, rank, M, floor)`` each
track is scored at.

Pinning the regime is what makes a tier reproducible and stops a submission from
cherry-picking the rank/M that flatters its method (the M-choice can flip a
verdict — a low-rank method that gates at M=48 can dominate at M=64). The GPU
bot scores every feat PR at *its declared track's* regime here, never the knobs
in the PR body. This module is the single source of truth for track floors,
consumed by ``eval.evaluator`` (the accuracy gate) and the GPU-test workflow
(the pinned run + the dashboard floors).

Standalone: no imports from the rest of ``eval`` (so evaluator can import it
without a cycle).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackSpec:
    """The pinned scoring regime for one track.

    fill           : eval ``--fill`` for this track's data.
    data_rank      : rank for lowrank / decaying-spectrum data (None => full-rank,
                     no low-rank structure to exploit).
    rank_m         : subspace dimension M to score at (None => ``default_rank_m(N)``,
                     i.e. the full-rank default that grows with N).
    accuracy_floor : accuracy below this hard-gates the score to 0 (whitepaper 6.2).
    """

    name: str
    fill: str
    data_rank: int | None
    rank_m: int | None
    accuracy_floor: float


#: The pinned regimes. Low-rank sits at M=64 = 4r (rank-16), the M where BOTH the
#: 3-way rsvd (needs 3r=48) and a 4-way method (needs 4r=64) fully recover — so
#: it's inclusive, not rigged toward one basis. Decaying-spectrum at rank-64/M=256.
TRACKS: dict[str, TrackSpec] = {
    "full-rank":         TrackSpec("full-rank",         "random",            None, None, 0.80),
    "low-rank":          TrackSpec("low-rank",          "lowrank",           16,   64,   0.95),
    "decaying-spectrum": TrackSpec("decaying-spectrum", "decaying-spectrum", 64,   256,  0.90),
}

DEFAULT_TRACK = "full-rank"

#: Which track a raw ``--fill`` belongs to (for resolving a run's floor from its
#: fill). ``iota`` is a full-rank-class synthetic fill.
_FILL_TO_TRACK: dict[str, str] = {spec.fill: name for name, spec in TRACKS.items()}
_FILL_TO_TRACK["iota"] = "full-rank"


def track_for_fill(fill: str) -> str:
    """Return the track name a ``fill`` is scored under (``full-rank`` default)."""
    return _FILL_TO_TRACK.get(fill, DEFAULT_TRACK)


def track_floor(fill: str) -> float:
    """The accuracy floor for ``fill``'s track."""
    return TRACKS[track_for_fill(fill)].accuracy_floor


def get_track(name: str) -> TrackSpec:
    """The pinned :class:`TrackSpec` for ``name`` (raises on an unknown track)."""
    if name not in TRACKS:
        raise KeyError(f"unknown track {name!r}; available: {sorted(TRACKS)}")
    return TRACKS[name]


def accuracy_floors() -> dict[str, float]:
    """``{track_name: floor}`` — for the dashboard's per-track floor display."""
    return {name: spec.accuracy_floor for name, spec in TRACKS.items()}
