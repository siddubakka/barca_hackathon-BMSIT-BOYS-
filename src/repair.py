"""Ball repair and low-level per-frame signals.

Inputs are the raw Track arrays; outputs are arrays the evidence module
consumes.  Nothing here consults labels or frame-specific constants.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .config import Config
from .constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from .load import Track


@dataclass
class Repaired:
    ball_x: np.ndarray  # (n,) repaired ball, NaN where unknown
    ball_y: np.ndarray
    ball_valid: np.ndarray  # (n,) bool: ball position known (seen or interpolated)
    ball_seen: np.ndarray  # (n,) bool: raw sighting this frame
    ball_gap: np.ndarray  # (n,) frames to nearest raw sighting (inf if none)
    ball_out: np.ndarray  # (n,) bool: known ball clearly outside the pitch
    ball_v: np.ndarray  # (n,) m/s, geometry-derived (NaN where unknowable)
    motion: np.ndarray  # (n,) m/s, median tracked-player speed
    motion_n: np.ndarray  # (n,) how many slots contributed to motion


def _fill_neighbours(mask: np.ndarray):
    """Return (prev_index, next_index): nearest True at/before and at/after each row.

    Values are original-array indices; -1 where no such neighbour exists.  Rows where
    ``mask`` is True map to their own index on both sides.
    """
    n = len(mask)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(mask, idx, -1))
    # next: the "previous True" trick on the reversed mask.  Accumulate the
    # rev-space *positions*, reverse back, then convert o = n-1-k to original
    # indices; -1 (no such neighbour) must survive the conversion.
    rev = mask[::-1]
    k_rev = np.maximum.accumulate(np.where(rev, idx, -1))[::-1]
    nxt = np.where(k_rev >= 0, n - 1 - k_rev, -1)
    nxt = np.where(mask, idx, nxt)
    return prev, nxt


def repair(tr: Track, cfg: Config) -> Repaired:
    n = tr.n
    seen = tr.seen_ball
    prev, nxt = _fill_neighbours(seen)

    # linear interpolation across short holes only ------------------------
    have_prev = prev >= 0
    have_next = nxt >= 0
    hole = np.where(have_prev & have_next, nxt - prev, np.iinfo(np.int64).max)
    interp_ok = have_prev & have_next & (hole <= cfg.max_hole_frames)

    ball_x = np.full(n, np.nan, dtype=np.float64)
    ball_y = np.full(n, np.nan, dtype=np.float64)

    # vectorised linear interpolation on the eligible rows
    rows = np.flatnonzero(interp_ok)
    if len(rows):
        p = prev[rows].astype(np.float64)
        q = nxt[rows].astype(np.float64)
        t = (rows - p) / np.maximum(q - p, 1.0)
        ball_x[rows] = tr.ball_x[p.astype(np.int64)] * (1 - t) + tr.ball_x[q.astype(np.int64)] * t
        ball_y[rows] = tr.ball_y[p.astype(np.int64)] * (1 - t) + tr.ball_y[q.astype(np.int64)] * t
    ball_x[seen] = tr.ball_x[seen]
    ball_y[seen] = tr.ball_y[seen]

    # distance (in frames) to the nearest raw sighting --------------------
    gap = np.full(n, np.inf)
    with np.errstate(invalid="ignore"):
        d_prev = np.where(have_prev, idx := np.arange(n) - prev, np.inf)
        d_next = np.where(have_next, nxt - np.arange(n), np.inf)
    gap = np.minimum(d_prev, d_next)
    gap[seen] = 0.0

    # ball clearly outside the pitch (tracked beyond the touchline) --------
    margin = 0.015
    ball_out = (
        seen
        & (
            (tr.ball_x < -margin)
            | (tr.ball_x > 1 + margin)
            | (tr.ball_y < -margin)
            | (tr.ball_y > 1 + margin)
        )
    )

    # geometry-derived ball speed (m/s), windows of up to 5 frames ---------
    ball_v = np.zeros(n, dtype=np.float64)
    seen_rows = np.flatnonzero(seen)
    L, W = tr.pitch
    if len(seen_rows) > 1:
        a = seen_rows[:-1]
        b = seen_rows[1:]
        dt = (b - a) / tr.fps
        close = dt <= (6.0 / tr.fps)  # pair only near-consecutive sightings
        a, b, dt = a[close], b[close], dt[close]
        dist = np.hypot(
            (tr.ball_x[b] - tr.ball_x[a]) * L,
            (tr.ball_y[b] - tr.ball_y[a]) * W,
        )
        v = dist / np.maximum(dt, 1e-6)
        # stamp speed onto both endpoints; missing in-between rows inherit
        # the pair speed via a short backward fill done below
        ball_v[a] = np.maximum(ball_v[a], v)
        ball_v[b] = np.maximum(ball_v[b], v)
        for lo, hi, val in zip(a, b, v):
            if hi - lo > 1:
                ball_v[lo : hi + 1] = np.maximum(ball_v[lo : hi + 1], val)
    # light rolling max so a kick stays visible for a couple of frames
    ball_v = _rolling_max(ball_v, 3)

    # motion: median speed of tracked players --------------------------------
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> 0 below
        finite = np.isfinite(tr.ps)
        motion_n = finite.sum(axis=1)
        motion = np.nanmedian(np.where(finite, tr.ps, np.nan), axis=1)
    motion = np.nan_to_num(motion, nan=0.0)
    motion[motion_n < 3] = 0.0  # too few players tracked -> treat as quiet

    return Repaired(
        ball_x=ball_x,
        ball_y=ball_y,
        ball_valid=np.isfinite(ball_x) & np.isfinite(ball_y),
        ball_seen=seen,
        ball_gap=gap,
        ball_out=ball_out,
        ball_v=ball_v,
        motion=motion,
        motion_n=motion_n,
    )


def _rolling_max(x: np.ndarray, half: int) -> np.ndarray:
    out = x.copy()
    for k in range(1, half + 1):
        shifted = np.full_like(x, -np.inf)
        shifted[k:] = x[:-k]
        out = np.maximum(out, shifted)
        shifted2 = np.full_like(x, -np.inf)
        shifted2[:-k] = x[k:]
        out = np.maximum(out, shifted2)
    return np.where(np.isfinite(out), out, 0.0)
