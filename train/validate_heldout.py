"""Generalization check: score the frozen configuration on held-out lab matches.

    python train/validate_heldout.py [--config train/sweep_results.json]

Tune matches are excluded; everything else under SKILLCORNER_DIR is evaluated.
Reports per-match and mean spellF1/macroF1 plus structure diagnostics (D share,
possessions per minute, D runs per minute) for the transfer sanity check in
docs/TECHNICAL_REPORT.md §7.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402
from train.lab import list_matches, load_lab_match, pm, run_pipeline  # noqa: E402


def structure(labels: np.ndarray) -> dict:
    """Segment-level structure of a prediction/GT array (labels 0..3)."""
    runs_mask = labels != 3
    n_min = max(1, int(runs_mask.sum()) / 25 / 60)
    # count constant runs inside non-X spans
    segs = []
    in_run = False
    start = 0
    for i, v in enumerate(labels):
        if v == 3:
            if in_run:
                segs.append((start, i, labels[start]))
                in_run = False
            continue
        if not in_run:
            in_run, start = True, i
        elif v != labels[i - 1]:
            segs.append((start, i, labels[i - 1]))
            start = i
    if in_run:
        segs.append((start, len(labels), labels[start]))
    poss = [s for s in segs if s[2] in (0, 1)]
    dsegs = [s for s in segs if s[2] == 2]
    d_frames = int((labels == 2)[runs_mask].sum())
    graded = int(runs_mask.sum())
    return {
        "poss_per_min": len(poss) / n_min,
        "d_runs_per_min": len(dsegs) / n_min,
        "d_share": d_frames / max(1, graded),
        "n_poss": len(poss),
        "n_d_runs": len(dsegs),
    }


# First-principles values the sweep started from (recorded for the
# defaults-vs-tuned comparison in the report).
FIRST_PRINCIPLES = dict(
    sigma_m=3.5, th_switch=0.25, th_motion_low=0.8, th_motion_high=1.7,
    th_ball_move=4.0, t_exit_frames=4, t_min_state_frames=15,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "train", "sweep_results.json"))
    ap.add_argument("--all", action="store_true", help="include tune matches in the table")
    ap.add_argument("--defaults", action="store_true",
                    help="score the first-principles config instead of the frozen one")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        payload = json.load(f)
    best = payload["best"]
    tune = set(payload["tune_matches"])
    if args.defaults:
        cfg = Config.from_dict({**best, **FIRST_PRINCIPLES})
        print(f"first-principles config: {json.dumps(cfg.to_dict())}")
    else:
        cfg = Config.from_dict(best)
        print(f"frozen config: {json.dumps(best)}")
    matches = [m for m in list_matches() if args.all or m not in tune]
    print(f"evaluating {len(matches)} {'(all)' if args.all else 'held-out'} matches\n")

    spells, macros, adfs = [], [], []
    rows = []
    for m in matches:
        tr, gt, tags, _ = load_lab_match(m)
        pred = run_pipeline(tr, tags, cfg)
        r = pm.evaluate(gt, pred)
        st_pred, st_gt = structure(pred), structure(gt)
        spells.append(r["spell_f1_mean"])
        macros.append(r["macro_f1"])
        adfs.append(r["alive_dead_f1"])
        rows.append((m, r, st_pred, st_gt))
        print(
            f"{m}  spell {r['spell_f1_mean']:.4f}  macro {r['macro_f1']:.4f}"
            f"  a/d {r['alive_dead_f1']:.3f}"
            f"  spells pred/gt {r['n_pred_spells']}/{r['n_gt_spells']}"
            f" | D% pred {st_pred['d_share']:.2f} gt {st_gt['d_share']:.2f}"
            f" | poss/min pred {st_pred['poss_per_min']:.1f} gt {st_gt['poss_per_min']:.1f}"
            f" | Druns/min pred {st_pred['d_runs_per_min']:.1f} gt {st_gt['d_runs_per_min']:.1f}",
            flush=True,
        )

    def mean(xs):
        return sum(xs) / len(xs)

    print(
        f"\nMEAN over {len(matches)}: spell {mean(spells):.4f}  "
        f"macro {mean(macros):.4f}  alive_dead {mean(adfs):.3f}"
    )
    print(
        "structure means: D% pred {:.2f} gt {:.2f} | poss/min pred {:.1f} gt {:.1f}"
        " | Druns/min pred {:.1f} gt {:.1f}".format(
            mean([r[2]["d_share"] for r in rows]),
            mean([r[3]["d_share"] for r in rows]),
            mean([r[2]["poss_per_min"] for r in rows]),
            mean([r[3]["poss_per_min"] for r in rows]),
            mean([r[2]["d_runs_per_min"] for r in rows]),
            mean([r[3]["d_runs_per_min"] for r in rows]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
