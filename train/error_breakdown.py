"""Decompose misses / spurious spells into causes (lab data only).

Pure diagnosis: nothing here selects or changes a configuration.

For every lab match, decode with the frozen config and ask, for every
GT spell we miss at IoU 0.3 (and every predicted spell that matches
nothing), *what were we actually saying* over that span:

  * dead majority       -> stoppage/timing error (we held D over alive ball)
  * wrong-team majority -> team attribution error
  * right-team majority -> right team, poor overlap (boundary/fragment error)
  * mixed               -> no single majority

and, symmetrically, for every spurious predicted spell, what the *ground
truth* says over its span.  Plus a restart-focused block: at every GT
D->alive transition, whether we are alive, dead or on the wrong team at
the exact restart frame, and how many frames until we agree.

Tune and held-out are aggregated separately (held-out is the number to
quote).  Usage::

    python train/error_breakdown.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402
from train.error_analysis import _pred_flags  # noqa: E402
from train.lab import list_matches, load_lab_match, pm, run_pipeline  # noqa: E402

FPS = 25
OUT_PATH = os.path.join(REPO_ROOT, "train", "error_breakdown_results.json")
IOU = 0.3
RESTART_SEARCH = 50  # frames (2 s) to wait for agreement after a GT restart


def _frac(labels: np.ndarray, a: int, b: int) -> tuple[float, float, float]:
    """(dead, team0, team1) frame shares of labels[a:b]."""
    window = labels[a:b]
    if not len(window):
        return 1.0, 0.0, 0.0
    n = float(len(window))
    return (
        float((window == 2).sum()) / n,
        float((window == 0).sum()) / n,
        float((window == 1).sum()) / n,
    )


def analyse(match_id: str, cfg: Config) -> dict:
    tr, gt, tags, _ = load_lab_match(match_id)
    pred = run_pipeline(tr, tags, cfg)
    runs = pm.valid_runs(gt)

    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    gflags = pm.matched_gt_flags(overlaps, len(gt_spells), IOU)
    pflags = _pred_flags(overlaps, len(pred_spells), IOU)

    # ---- missed GT spells -> what we said over their span -----------------
    miss_cat = {"dead": 0, "wrong_team": 0, "right_team": 0, "mixed": 0}
    miss_fracs = []
    for i, (a, b, k) in enumerate(gt_spells):
        if gflags[i]:
            continue
        d = _frac(pred, a, b)[0]
        right = _frac(pred, a, b)[1 + k]
        wrong = _frac(pred, a, b)[2 - k]
        miss_fracs.append((d, right, wrong))
        vals = {"dead": d, "wrong_team": wrong, "right_team": right}
        top = max(vals, key=vals.get)
        if vals[top] >= 0.5:
            miss_cat[top] += 1
        else:
            miss_cat["mixed"] += 1

    # ---- spurious predicted spells -> what GT says over their span -------
    spur_cat = {"gt_dead": 0, "wrong_team": 0, "right_team": 0, "mixed": 0}
    spur_fracs = []
    for j, (a, b, k) in enumerate(pred_spells):
        if pflags[j]:
            continue
        d = _frac(gt, a, b)[0]
        right = _frac(gt, a, b)[1 + k]
        wrong = _frac(gt, a, b)[2 - k]
        spur_fracs.append((d, right, wrong))
        vals = {"gt_dead": d, "wrong_team": wrong, "right_team": right}
        top = max(vals, key=vals.get)
        if vals[top] >= 0.5:
            spur_cat[top] += 1
        else:
            spur_cat["mixed"] += 1

    # ---- restart block: GT D->alive transitions ---------------------------
    segs = pm.segments(gt, runs)
    restart_states = {"alive_correct": 0, "alive_wrong_team": 0, "still_dead": 0}
    frames_to_agree = []
    early_exit = 0  # we were alive while GT still dead just before the restart
    n_restarts = 0
    for idx, (a, b, k) in enumerate(segs):
        if k != 2 or idx + 1 >= len(segs):
            continue
        na, nb, nk = segs[idx + 1]
        if nk == 2 or na != b:
            continue  # dead run not directly followed by alive
        n_restarts += 1
        if pred[b - 1] != 2:
            early_exit += 1
        p = pred[b]
        if p == 2:
            restart_states["still_dead"] += 1
        elif p == nk:
            restart_states["alive_correct"] += 1
        else:
            restart_states["alive_wrong_team"] += 1
        # frames until pred agrees (alive + right team)
        limit = min(b + RESTART_SEARCH, nb)
        agree = next((t for t in range(b, limit) if pred[t] == nk), None)
        if agree is not None:
            frames_to_agree.append(agree - b)
        else:
            frames_to_agree.append(RESTART_SEARCH)  # censored at 2 s

    # ---- structure -------------------------------------------------------
    graded = gt != 3
    pred_d = float((pred[graded] == 2).mean()) if graded.any() else 0.0
    gt_d = float((gt[graded] == 2).mean()) if graded.any() else 0.0

    return {
        "n_missed": sum(miss_cat.values()),
        "miss_cat": miss_cat,
        "miss_fracs": miss_fracs,
        "n_spurious": sum(spur_cat.values()),
        "spur_cat": spur_cat,
        "spur_fracs": spur_fracs,
        "n_restarts": n_restarts,
        "restart_states": restart_states,
        "early_exit": early_exit,
        "frames_to_agree": frames_to_agree,
        "pred_d_share": pred_d,
        "gt_d_share": gt_d,
        "spell": pm.evaluate(gt, pred)["spell_f1_mean"],
    }


def _mean_fracs(rows: list[dict], key: str) -> list[float]:
    xs = [x for r in rows for x in r[key]]
    if not xs:
        return [0.0, 0.0, 0.0]
    return [round(float(np.mean([x[i] for x in xs])), 4) for i in range(3)]


def _pct_cat(rows: list[dict], key: str) -> dict:
    cat = {}
    for r in rows:
        for k, v in r[key].items():
            cat[k] = cat.get(k, 0) + v
    total = sum(cat.values()) or 1
    return {k: round(100.0 * v / total, 1) for k, v in sorted(cat.items())}


def _agg(rows: list[dict]) -> dict:
    agree = [x for r in rows for x in r["frames_to_agree"]]
    states = {}
    for r in rows:
        for k, v in r["restart_states"].items():
            states[k] = states.get(k, 0) + v
    n_states = sum(states.values()) or 1
    n_rest = sum(r["n_restarts"] for r in rows) or 1
    return {
        "n_matches": len(rows),
        "spell_mean": round(float(np.mean([r["spell"] for r in rows])), 4),
        "missed_total": sum(r["n_missed"] for r in rows),
        "miss_causes_pct": _pct_cat(rows, "miss_cat"),
        "miss_mean_fracs_d_right_wrong": _mean_fracs(rows, "miss_fracs"),
        "spurious_total": sum(r["n_spurious"] for r in rows),
        "spurious_causes_pct": _pct_cat(rows, "spur_cat"),
        "spurious_mean_fracs_gtdead_gtright_gtwrong": _mean_fracs(rows, "spur_fracs"),
        "restarts": n_rest,
        "restart_state_pct": {
            k: round(100.0 * v / n_states, 1) for k, v in sorted(states.items())
        },
        "early_exit_pct": round(
            100.0 * sum(r["early_exit"] for r in rows) / n_rest, 1),
        "restart_frames_to_agree_mean": round(float(np.mean(agree)), 1)
        if agree else None,
        "restart_agree_within_12f_pct": round(
            100.0 * float(np.mean([x <= 12 for x in agree])), 1) if agree else None,
        "pred_d_share": round(
            float(np.mean([r["pred_d_share"] for r in rows])), 4),
        "gt_d_share": round(float(np.mean([r["gt_d_share"] for r in rows])), 4),
    }


def main() -> None:
    with open(os.path.join(REPO_ROOT, "train", "sweep_results.json"),
              encoding="utf-8") as f:
        tune = set(json.load(f)["tune_matches"])

    cfg = Config()
    ids = list_matches()
    print(f"{len(ids)} lab matches; tune split: {sorted(tune)}", flush=True)
    rows_tune, rows_held = [], []
    for mid in ids:
        r = analyse(mid, cfg)
        (rows_tune if mid in tune else rows_held).append(r)
        print(
            f"  {mid}: spell {r['spell']:.4f} missed {r['n_missed']} "
            f"spurious {r['n_spurious']} restarts {r['n_restarts']}",
            flush=True,
        )

    out = {"frozen": {"tune": _agg(rows_tune), "heldout": _agg(rows_held)}}
    for split in ("heldout", "tune"):
        a = out["frozen"][split]
        print(f"\n=== {split} ===")
        print(f"spell mean            {a['spell_mean']}")
        print(f"missed total          {a['missed_total']}")
        print(f"  causes %            {a['miss_causes_pct']}")
        print(f"  mean fracs (D,right,wrong) {a['miss_mean_fracs_d_right_wrong']}")
        print(f"spurious total        {a['spurious_total']}")
        print(f"  causes %            {a['spurious_causes_pct']}")
        print(f"  mean fracs (GTdead, GTright, GTwrong) "
              f"{a['spurious_mean_fracs_gtdead_gtright_gtwrong']}")
        print(f"restarts              {a['restarts']}")
        print(f"  state at restart %  {a['restart_state_pct']}")
        print(f"  early exit %        {a['early_exit_pct']}")
        print(f"  frames to agree     mean {a['restart_frames_to_agree_mean']} "
              f"within 12f {a['restart_agree_within_12f_pct']}%")
        print(f"D share pred/gt       {a['pred_d_share']}/{a['gt_d_share']}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
