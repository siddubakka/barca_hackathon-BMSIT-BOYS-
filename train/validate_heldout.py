"""Generalization check: score a configuration on held-out lab matches.

    python train/validate_heldout.py [--config train/sweep2_results.json]
                                     [--defaults] [--all]

Tune matches are excluded; everything else under SKILLCORNER_DIR is
evaluated.  The frozen configuration (Config()) is evaluated in the same
pass so every printed row is a paired comparison against it (mean delta and
per-match wins) -- this is the single held-out run used by the adoption bar
documented in train/sweep3.py.

Structure diagnostics (D share, possessions per minute, D runs per minute)
are computed on the *graded* frames of each match (gt != 3) for BOTH the
prediction and the GT, so pred and GT columns are directly comparable; an
earlier version measured the prediction over all frames, which inflated its
D share with ungraded (halftime) D and deflated its per-minute rates.
Reported in docs/TECHNICAL_REPORT.md section 7.
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


def structure(labels: np.ndarray, graded: np.ndarray | None = None) -> dict:
    """Segment-level structure of a labels array (0..3) on graded frames.

    `graded` is the boolean mask of frames that carry a GT label; frames
    outside it are treated as X so prediction and GT share one denominator.
    """
    if graded is not None:
        labels = np.where(graded, labels, 3)
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
    graded_n = int(runs_mask.sum())
    return {
        "poss_per_min": len(poss) / n_min,
        "d_runs_per_min": len(dsegs) / n_min,
        "d_share": d_frames / max(1, graded_n),
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
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the paired frozen-config baseline (diagnostic only)")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        payload = json.load(f)
    if "best" in payload:                      # sweep_results / sweep2 shape
        best = payload["best"]
    else:                                      # sweep3_results.json shape:
        best = payload["selected"]["config"]   # the pre-registered selection
    if "tune_matches" in payload:
        tune = set(payload["tune_matches"])
    else:
        with open(os.path.join(REPO_ROOT, "train", "sweep_results.json"),
                  encoding="utf-8") as f:
            tune = set(json.load(f)["tune_matches"])
    if args.defaults:
        cfg = Config.from_dict({**best, **FIRST_PRINCIPLES})
        print(f"first-principles config: {json.dumps(cfg.to_dict())}")
    else:
        cfg = Config.from_dict(best)
        print(f"candidate config: {json.dumps(best)}")
    base_cfg = Config()  # frozen reference, evaluated in the same pass
    if not args.no_baseline:
        print(f"frozen baseline:  {json.dumps(base_cfg.to_dict())}")
    matches = [m for m in list_matches() if args.all or m not in tune]
    print(f"evaluating {len(matches)} {'(all)' if args.all else 'held-out'} matches\n")

    spells, macros, adfs = [], [], []
    b_spells, b_macros, b_adfs = [], [], []
    rows = []
    for m in matches:
        tr, gt, tags, _ = load_lab_match(m)
        graded = gt != 3
        pred = run_pipeline(tr, tags, cfg)
        r = pm.evaluate(gt, pred)
        st_pred, st_gt = structure(pred, graded), structure(gt)
        spells.append(r["spell_f1_mean"])
        macros.append(r["macro_f1"])
        adfs.append(r["alive_dead_f1"])
        line = (
            f"{m}  spell {r['spell_f1_mean']:.4f}  macro {r['macro_f1']:.4f}"
            f"  a/d {r['alive_dead_f1']:.3f}"
            f"  spells pred/gt {r['n_pred_spells']}/{r['n_gt_spells']}"
            f" | D% pred {st_pred['d_share']:.2f} gt {st_gt['d_share']:.2f}"
            f" | poss/min pred {st_pred['poss_per_min']:.1f} gt {st_gt['poss_per_min']:.1f}"
            f" | Druns/min pred {st_pred['d_runs_per_min']:.1f} gt {st_gt['d_runs_per_min']:.1f}"
        )
        if not args.no_baseline:
            pred_b = run_pipeline(tr, tags, base_cfg)
            rb = pm.evaluate(gt, pred_b)
            st_b = structure(pred_b, graded)
            b_spells.append(rb["spell_f1_mean"])
            b_macros.append(rb["macro_f1"])
            b_adfs.append(rb["alive_dead_f1"])
            rows.append((m, r, st_pred, st_gt, rb, st_b))
            line += (f"\n{'':2}vs frozen: spell {rb['spell_f1_mean']:.4f} "
                     f"delta {r['spell_f1_mean'] - rb['spell_f1_mean']:+.4f}  "
                     f"macro {rb['macro_f1']:.4f} "
                     f"delta {r['macro_f1'] - rb['macro_f1']:+.4f}"
                     f" | D% {st_b['d_share']:.2f} poss/min "
                     f"{st_b['poss_per_min']:.1f} Druns {st_b['d_runs_per_min']:.1f}")
        else:
            rows.append((m, r, st_pred, st_gt))
        print(line, flush=True)

    def mean(xs):
        return sum(xs) / len(xs)

    print(
        f"\nMEAN over {len(matches)}: spell {mean(spells):.4f}  "
        f"macro {mean(macros):.4f}  alive_dead {mean(adfs):.3f}"
    )
    print(
        "structure means (graded frames): D% pred {:.2f} gt {:.2f}"
        " | poss/min pred {:.1f} gt {:.1f}"
        " | Druns/min pred {:.1f} gt {:.1f}".format(
            mean([r[2]["d_share"] for r in rows]),
            mean([r[3]["d_share"] for r in rows]),
            mean([r[2]["poss_per_min"] for r in rows]),
            mean([r[3]["poss_per_min"] for r in rows]),
            mean([r[2]["d_runs_per_min"] for r in rows]),
            mean([r[3]["d_runs_per_min"] for r in rows]),
        )
    )
    if b_spells:
        wins = sum(1 for a, b in zip(spells, b_spells) if a > b)
        print(
            f"\nvs FROZEN: spell {mean(b_spells):.4f} -> {mean(spells):.4f} "
            f"(delta {mean(spells) - mean(b_spells):+.4f})  "
            f"macro {mean(b_macros):.4f} -> {mean(macros):.4f} "
            f"(delta {mean(macros) - mean(b_macros):+.4f})  "
            f"alive_dead {mean(b_adfs):.3f} -> {mean(adfs):.3f}  "
            f"per-match wins {wins}/{len(matches)}"
        )
        print(
            "frozen structure means (graded frames): D% {:.2f} | "
            "poss/min {:.1f} | Druns/min {:.1f}".format(
                mean([r[5]["d_share"] for r in rows]),
                mean([r[5]["poss_per_min"] for r in rows]),
                mean([r[5]["d_runs_per_min"] for r in rows]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
