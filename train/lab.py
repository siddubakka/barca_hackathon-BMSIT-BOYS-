"""SkillCorner practice lab: tune the decoder on other matches' labels.

Why this exists
---------------
The competition's ground truth may not be used for tuning (only the pipeline
may be checked against the public 5-minute window once, at the end).  The
playbook's answer is a *practice lab*: labeled matches from a public dataset,
run through the exact same pipeline code, so every hyperparameter in
``src/config.py`` is selected on legal data.

Data: SkillCorner open-data matches (10 fps broadcast tracking + annotations).
One match folder contains::

    {id}_tracking_extrapolated.jsonl   per-frame ball/players/possession
    {id}_phases_of_play.csv            possession spells with the team id
    {id}_dynamic_events.csv            (not used: no set-piece events)
    {id}_match.json                    teams, pitch, periods

Lab ground truth (documented, deterministic, derived only from SkillCorner's
own labels -- nothing from the competition match):

    outside a match period            -> X  (halftime / pre-match, ungraded)
    inside a possession phase         -> H/A from ``team_in_possession_id``
    between phases (a gap)            -> D, unless SkillCorner's per-frame
                                         ``possession`` names a team (loose
                                         ball they could still attribute),
                                         which then wins over the gap.

Why this approximates the competition GT: in-phase frames have the ball
detected 94% of the time and the ball almost never outside the pitch (0.8%);
gap frames have no possessor 95% of the time and the ball outside the pitch
44% of the time.  Phases are the "ball alive, team controls" spans; gaps are
"not in play / not controlled", which is the competition's D semantics.

Pipeline identity
-----------------
To make the lab exercise *literally the same code*, matches are resampled
10 fps -> 25 fps and wrapped in the competition's ``Track`` structure:

  * ball positions resampled; in-play ball visibility degraded to 65%
    (the competition's estimated in-play visibility -- SkillCorner's own is
    ~90%, far cleaner than the broadcast data we must survive);
  * player speeds derived from positions at 10 fps (SkillCorner ships no
    speed channel), temporally smoothed, then resampled -- the competition's
    ATD ships a real speed channel, so this is the closest legal stand-in;
  * synthetic ``SET PIECES`` restart tags built from the GT dead-ball runs
    with +-0.5 s edge jitter, bracketing the restart moment (the real tags
    are mostly fixed 35 s windows covering the whole stoppage; the lab's
    ~5 s brackets cover the pre-kick dead time the decoder needs and are
    never perfectly aligned).  They carry no team label (throw-ins, the
    majority of real set pieces, are N/A), so no team information leaks;
  * ``w_tag`` (phase-team prior) is therefore NOT tunable here and keeps its
    first-principles default -- only tracking-driven parameters are tuned.

No parameter is ever selected on competition data or on the public GT.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402
from src.decode import decode  # noqa: E402
from src.evidence import compute  # noqa: E402
from src.load import Tag, Track  # noqa: E402
from src.repair import repair  # noqa: E402

# Official scorer (read-only, from the challenge repository).  Only the lab,
# sweep and validation scripts need it -- regenerating the submission never
# does -- so it is resolved from $MTAH_PM_DIR or a sibling clone of the
# challenge repository, with a clear error if neither is present.
def _pm_dir() -> str:
    cands = [
        os.environ.get("MTAH_PM_DIR", ""),
        os.path.join(os.path.dirname(REPO_ROOT), "more-than-a-hack-2027",
                     "evaluation"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "possession_metrics.py")):
            return c
    raise ImportError(
        "possession_metrics.py not found.  Set MTAH_PM_DIR to the challenge "
        "repository's evaluation/ directory.  Only the lab/sweep scripts "
        "need it; make_submission.py runs without it."
    )


PM_DIR = _pm_dir()
if PM_DIR not in sys.path:
    sys.path.insert(0, PM_DIR)
import possession_metrics as pm  # noqa: E402

# SkillCorner lab data and its parse cache.  Both live outside this
# repository (competition rule: no data committed) and are located via
# environment variables, so a fresh clone carries no machine-specific paths:
#   SKILLCORNER_DIR  directory holding the {id}_* SkillCorner match files
#   LAB_CACHE_DIR    writable directory for the parsed .npz caches
# Lab scripts only -- make_submission.py never touches these.
SC_ROOT = os.environ.get("SKILLCORNER_DIR", "")
CACHE_ROOT = os.environ.get("LAB_CACHE_DIR", "")

SRC_FPS = 10
DST_FPS = 25
RATIO = DST_FPS / SRC_FPS  # 2.5

# Estimated share of in-play frames where the competition's ball tracking is
# visible (graded ball visibility 59.3% with ~20-25% dead time, most of which
# also hides the ball).  SkillCorner's own in-play visibility is ~90%, so the
# lab drops detections down to this level to match the noise the pipeline must
# actually survive.
LAB_INPLAY_BALL_VISIBILITY = 0.65

CACHE_VERSION = "v2"  # bump when GT/degradation logic changes

HOME, AWAY, DEAD, ABSTAIN = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# parsing (once per match, then cached)
# --------------------------------------------------------------------------- #

def _parse_match(match_id: str) -> dict:
    """SkillCorner files -> 10 fps arrays + GT + slot metadata."""
    base = os.path.join(SC_ROOT, str(match_id))
    with open(os.path.join(base, f"{match_id}_match.json"), encoding="utf-8") as f:
        meta = json.load(f)

    home_id = str(meta["home_team"]["id"])
    pitch = (float(meta["pitch_length"]), float(meta["pitch_width"]))
    players = meta["players"]
    slot_of_pid = {str(p["id"]): i for i, p in enumerate(players)}
    team_of_slot = np.array(
        [0 if str(p.get("team_id")) == home_id else 1 for p in players], dtype=np.int8
    )
    n_slots = len(players)

    # ---- tracking --------------------------------------------------------
    rows = []
    with open(
        os.path.join(base, f"{match_id}_tracking_extrapolated.jsonl"), encoding="utf-8"
    ) as f:
        for line in f:
            d = json.loads(line)
            b = d.get("ball_data") or {}
            p = d.get("possession") or {}
            bx = b.get("x")
            by = b.get("y")
            det = b.get("is_detected") is True and bx is not None and by is not None
            grp = p.get("group")
            rows.append(
                (
                    det,
                    float(bx) if det else np.nan,
                    float(by) if det else np.nan,
                    0 if grp == "home team" else (1 if grp == "away team" else -1),
                    int(d.get("period") or 0),
                    [(pd_["player_id"], pd_.get("x"), pd_.get("y"))
                     for pd_ in (d.get("player_data") or [])],
                )
            )

    n = len(rows)
    ball_x = np.array([r[1] for r in rows], dtype=np.float32)
    ball_y = np.array([r[2] for r in rows], dtype=np.float32)
    seen = np.array([r[0] for r in rows], dtype=bool)
    poss = np.array([r[3] for r in rows], dtype=np.int8)
    period = np.array([r[4] for r in rows], dtype=np.int8)
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for pid, x, y in r[5]:
            s = slot_of_pid.get(str(pid))
            if s is not None and x is not None and y is not None:
                px[i, s] = x
                py[i, s] = y

    # ---- ground truth at 10 fps ------------------------------------------
    gt = np.full(n, ABSTAIN, dtype=np.int8)
    gt[period > 0] = DEAD  # in a period: default = not in play
    ph_path = os.path.join(base, f"{match_id}_phases_of_play.csv")
    with open(ph_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = str(row["team_in_possession_id"])
            team = 0 if tid == home_id else 1
            a = max(0, int(row["frame_start"]))
            b = min(n, int(row["frame_end"]))
            if b > a:
                gt[a:b] = team
    # gaps where SkillCorner still attributes a possessor -> that team wins
    loose = (gt == DEAD) & (poss >= 0)
    gt[loose] = poss[loose]

    # ---- player speed (m/s) from positions, smoothed at 10 fps -----------
    ps = np.full((n, n_slots), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        dx = np.diff(px, axis=0, prepend=px[:1])
        dy = np.diff(py, axis=0, prepend=py[:1])
        step = np.hypot(dx, dy)
        v = step * SRC_FPS
        # only between consecutive frames where both endpoints are present
        both = np.isfinite(px) & np.isfinite(np.vstack([px[:1], px[:-1]]))
        v[~both] = np.nan
        v[0] = np.nan
        # a player teleporting across the pitch is a tracking swap, not a run
        v[v > 12.0] = np.nan
    for s in range(n_slots):
        col = pd.Series(v[:, s])
        ps[:, s] = (
            col.rolling(5, center=True, min_periods=1).median().to_numpy(dtype=np.float32)
        )

    # ---- degrade in-play ball visibility to the competition's level -------
    inplay = (gt == HOME) | (gt == AWAY)
    rng = np.random.default_rng(int(match_id) & 0xFFFFFFFF)
    target_seen = int(round(LAB_INPLAY_BALL_VISIBILITY * int(inplay.sum())))
    need = int((seen & inplay).sum()) - target_seen
    guard = 0
    while need > 0 and guard < 20000:
        guard += 1
        u = rng.random()
        if u < 0.55:
            length = int(rng.integers(2, 11))
        elif u < 0.85:
            length = int(rng.integers(11, 41))
        else:
            length = int(rng.integers(41, 101))
        lo = int(rng.integers(0, n))
        hi = min(n, lo + length)
        idx = np.arange(lo, hi)
        drop = seen[idx] & inplay[idx]
        if drop.any():
            seen[idx] = np.where(drop, False, seen[idx])
            need -= int(drop.sum())

    ball_x = np.where(seen, ball_x, np.nan)
    ball_y = np.where(seen, ball_y, np.nan)

    return {
        "n": n,
        "ball_x10": ball_x,
        "ball_y10": ball_y,
        "px10": px,
        "py10": py,
        "ps10": ps,
        "gt10": gt,
        "period10": period,
        "team_of_slot": team_of_slot,
        "pitch": pitch,
        "team_names": (
            str(meta["home_team"].get("short_name") or meta["home_team"]["name"]),
            str(meta["away_team"].get("short_name") or meta["away_team"]["name"]),
        ),
        "home_id": home_id,
    }


def _cache_path(match_id: str) -> str:
    if not CACHE_ROOT:
        return ""  # no cache configured: parse in memory, save nothing
    return os.path.join(CACHE_ROOT, f"{match_id}_{CACHE_VERSION}.npz")


def _load_or_parse(match_id: str) -> dict:
    if not SC_ROOT:
        raise RuntimeError(
            "SKILLCORNER_DIR is not set -- point it at the SkillCorner "
            "matches directory.  Lab scripts only; make_submission.py "
            "does not need it."
        )
    path = _cache_path(match_id)
    if path and os.path.isfile(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["version"]) == CACHE_VERSION:
                out = {k: z[k] for k in z.files if k != "version"}
                out["team_names"] = tuple(str(x) for x in out["team_names"])
                out["pitch"] = (float(out["pitch"][0]), float(out["pitch"][1]))
                out["home_id"] = str(out["home_id"])
                return out
        except Exception:
            pass
    data = _parse_match(match_id)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, version=CACHE_VERSION, **data)
    return data


# --------------------------------------------------------------------------- #
# resample to the competition's frame grid and wrap as a Track
# --------------------------------------------------------------------------- #

def _restart_tags(gt25: np.ndarray, match_id: str) -> list[Tag]:
    """Synthetic SET PIECES windows bracketing each dead-ball run's restart.

    Real SET PIECES tags are mostly fixed 35 s windows covering the whole
    stoppage (measured on the competition match: 59 tags, median 35 s,
    shortest 5 s).  We bracket the *end* of each lab dead run with a ~5 s
    window, +-0.5 s edge jitter, so the lab never hands the decoder a
    perfectly aligned restart marker.  Short gaps (< 4 s) are loose-ball
    phases rather than stoppages and get no tag.
    """
    rng = np.random.default_rng((int(match_id) * 2654435761) & 0xFFFFFFFF)
    tags: list[Tag] = []
    n = len(gt25)
    is_d = gt25 == DEAD
    edges = np.diff(np.concatenate(([0], is_d.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for a, b in zip(starts, stops):
        if (b - a) < 4 * DST_FPS:
            continue
        j0 = int(rng.integers(-13, 14))  # +-0.52 s
        j1 = int(rng.integers(-13, 14))
        start = int(np.clip(b - int(3.5 * DST_FPS) + j0, 0, n))
        end = int(np.clip(b + int(1.5 * DST_FPS) + j1, 0, n))
        if end > start:
            tags.append(Tag(code="SET PIECES", start_f=start, end_f=end, tags={}))
    return tags


def _blocks_from_period(period: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Contiguous runs where ``period > 0`` as half-open (start, end) blocks.

    Frames outside a period (pre-match, the halftime hole) fall between
    blocks, so the decoder emits D there and resets its state at each block
    start -- exactly what it does on the competition match, where the two
    graded play blocks are separate decode calls.
    """
    inside = (period > 0).astype(np.int8)
    edges = np.diff(np.concatenate(([0], inside, [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return tuple((int(a), int(b)) for a, b in zip(starts, stops))


def load_lab_match(match_id: str) -> tuple[Track, np.ndarray, list[Tag], dict]:
    """Return (Track at 25 fps, gt codes 0..3, synthetic tags, stats)."""
    d = _load_or_parse(str(match_id))
    n10 = int(d["n"])
    idx = np.minimum((np.arange(int(np.ceil(n10 * RATIO))) / RATIO).astype(int), n10 - 1)
    n = len(idx)

    L, W = d["pitch"]

    def norm(a: np.ndarray) -> np.ndarray:
        return (a / L + 0.5).astype(np.float32)

    blocks = _blocks_from_period(d["period10"][idx])
    tr = Track(
        n=n,
        fps=DST_FPS,
        ball_x=norm(d["ball_x10"][idx]),
        ball_y=(d["ball_y10"][idx] / W + 0.5).astype(np.float32),
        ball_s=np.full(n, np.nan, dtype=np.float32),
        px=(d["px10"][idx] / L + 0.5).astype(np.float32),
        py=(d["py10"][idx] / W + 0.5).astype(np.float32),
        ps=d["ps10"][idx].astype(np.float32),
        team=d["team_of_slot"],
        team_names=d["team_names"],
        pitch=(L, W),
        meta={"match_id": str(match_id), "blocks": blocks},
    )
    gt = d["gt10"][idx].astype(np.int8)
    tags = _restart_tags(gt, str(match_id))

    counts = np.bincount(gt, minlength=4)
    stats = {
        "n": n,
        "gt_counts": {
            "H": int(counts[HOME]), "A": int(counts[AWAY]),
            "D": int(counts[DEAD]), "X": int(counts[ABSTAIN]),
        },
        "n_tags": len(tags),
        "n_blocks": len(blocks),
        "ball_seen_pct": round(100 * float(np.isfinite(tr.ball_x).mean()), 1),
        "inplay_ball_seen_pct": round(
            100 * float(np.isfinite(tr.ball_x)[(gt == HOME) | (gt == AWAY)].mean()), 1
        ),
    }
    return tr, gt, tags, stats


# --------------------------------------------------------------------------- #
# run the pipeline and score
# --------------------------------------------------------------------------- #

def run_pipeline(tr: Track, tags: list[Tag], cfg: Config) -> np.ndarray:
    rep = repair(tr, cfg)
    ev = compute(tr, rep, tags, cfg)
    blocks = tr.meta.get("blocks") or ((0, tr.n),)
    return decode(tr, rep, ev, cfg, n=tr.n, active_blocks=blocks)


def score_match(match_id: str, cfg: Config, detail: bool = False) -> dict:
    tr, gt, tags, stats = load_lab_match(match_id)
    pred = run_pipeline(tr, tags, cfg)
    res = pm.evaluate(gt, pred)
    res["_stats"] = stats
    if detail:
        res["_pred"] = pred
        res["_gt"] = gt
    return res


def list_matches() -> list[str]:
    if not SC_ROOT:
        return []  # lab data not configured; lab scripts report "0 matches"
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(SC_ROOT, "*"))
        if os.path.isdir(p)
    )


if __name__ == "__main__":
    ids = list_matches()
    print(f"{len(ids)} lab matches under {SC_ROOT}")
    cfg = Config()
    for mid in ids[:3]:
        tr, gt, tags, stats = load_lab_match(mid)
        print(mid, stats)
        r = score_match(mid, cfg)
        print(
            f"  default cfg: spellF1 {r['spell_f1_mean']:.4f}  macro {r['macro_f1']:.4f}"
            f"  gt_spells {r['n_gt_spells']} pred {r['n_pred_spells']}"
            f"  alive_dead {r['alive_dead_f1']:.3f}"
        )
