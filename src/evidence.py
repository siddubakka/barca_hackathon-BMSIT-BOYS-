"""Per-frame evidence: who has the ball, and is it in play.

Two independent sources are combined here:

  1. Tracking vote -- a Gaussian proximity vote of every tracked player
     against the (repaired) ball position, summed per team.  This is a
     smoothed version of the baseline's nearest-player rule: instead of a
     hard argmin (which flips whenever a defender steps closer), nearby
     players contribute weight, so a defender screening the ball does not
     instantly steal the label while the opponent still controls it.

  2. Smart Tagging -- the provided SportsCode annotations.  Set-piece
     instances define restart windows (ball dead until the kick); phase
     instances carry a Team label whose meaning depends on the code
     (see OWN_POSSESSION / DEFENDING below).  Tag prior only *biases* the
     tracking vote (weight ``cfg.w_tag``); it never decides alone, because
     tags are ~20 s windows and can be stale at possession transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .load import Tag, Track
from .repair import Repaired

# Codes whose Team label is the team IN POSSESSION (attacking perspective).
OWN_POSSESSION = {
    "PROGRESSION",
    "BUILD UP",
    "LONG BALL",
    "CREATING CHANCES",
    "ATTACKING TRANSITION",
    "SET PIECES",  # the team taking the restart gains the ball at the kick
}
# Codes whose Team label is the team WITHOUT the ball (they are defending).
DEFENDING = {
    "DEFENSIVE TRANSITION",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
}
# Deliberately unused (semantics ambiguous from the doc alone):
#   BALL IN FINAL THIRD / BALL IN THE BOX / PLAYERS IN THE BOX / GOALS


@dataclass
class Evidence:
    e_team: np.ndarray  # (n,) tracking vote in [-1, 1]; NaN where ball unknown
    tag_score: np.ndarray  # (n,) weak phase prior in [-1, 1]; 0 where none
    in_restart: np.ndarray  # (n,) bool: inside a SET PIECES window
    n_near: np.ndarray  # (n,) players voting (within r_max)


def _team_sign(name: str, home: str, away: str) -> int:
    """+1 for the home team label, -1 for away, 0 for N/A or unknown."""
    n = (name or "").strip().lower()
    if not n or n == "n/a":
        return 0
    if n == home.strip().lower():
        return 1
    if n == away.strip().lower() or n.startswith("atl"):  # 'Atletic'/'Athletic'
        return -1
    return 0


def compute(tr: Track, rep: Repaired, tags: list[Tag], cfg: Config) -> Evidence:
    n = tr.n
    L, W = tr.pitch

    # ---- tracking vote ---------------------------------------------------
    e_team = np.full(n, np.nan, dtype=np.float64)
    n_near = np.zeros(n, dtype=np.int32)

    valid = rep.ball_valid
    home_mask = tr.team == 0
    away_mask = tr.team == 1
    w = np.zeros((n, tr.n_slots), dtype=np.float32)
    if valid.any():
        rows = np.flatnonzero(valid)
        bx = rep.ball_x[rows][:, None]
        by = rep.ball_y[rows][:, None]
        px = tr.px[rows].astype(np.float64)
        py = tr.py[rows].astype(np.float64)
        d2 = ((px - bx) * L) ** 2 + ((py - by) * W) ** 2
        finite = np.isfinite(d2)
        near = finite & (d2 <= cfg.r_max_m**2)
        g = np.zeros_like(d2)
        g[near] = np.exp(-d2[near] / (2.0 * cfg.sigma_m**2))
        w[rows] = g.astype(np.float32)
        n_near[rows] = near.sum(axis=1)

    home_w = w[:, home_mask].sum(axis=1, dtype=np.float64)
    away_w = w[:, away_mask].sum(axis=1, dtype=np.float64)
    tot = home_w + away_w
    ok = valid & (tot > 1e-9)
    e_team[ok] = (home_w[ok] - away_w[ok]) / tot[ok]

    # ---- Smart Tagging priors ---------------------------------------------
    home_name, away_name = tr.team_names
    tag_score = np.zeros(n, dtype=np.float64)
    in_restart = np.zeros(n, dtype=np.bool_)
    for t in tags:
        a = max(0, min(t.start_f, n))
        b = max(0, min(t.end_f, n))
        if b <= a:
            continue
        if t.code == "SET PIECES":
            in_restart[a:b] = True
        if t.code in OWN_POSSESSION:
            c = _team_sign(t.tags.get("team", ""), home_name, away_name)
        elif t.code in DEFENDING:
            c = -_team_sign(t.tags.get("team", ""), home_name, away_name)
        else:
            continue
        if c:
            tag_score[a:b] += c
    np.clip(tag_score, -1.0, 1.0, out=tag_score)

    return Evidence(e_team=e_team, tag_score=tag_score, in_restart=in_restart, n_near=n_near)
