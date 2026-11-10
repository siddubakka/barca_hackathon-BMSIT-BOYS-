"""Failure-mode decomposition of the frozen decoder (lab data only).

Answers the technical document's required "where does it fail, and what
would you fix with more time" question with measurements instead of
adjectives, and hunts for any *large* structural weakness.  Pure diagnosis:
nothing here selects or changes a configuration.

For every lab match, both the frozen config and the first-principles
defaults are decoded once and scored with the official scorer's own
functions (``possession_spells`` / ``spell_overlaps`` / ``matched_gt_flags``
-- no reimplementation), then broken down into:

  * spell recall/precision at IoU 0.3 and 0.5 (``spell_scores``),
  * pieces per ground-truth spell -- how often one true possession is split
    into several predicted ones (any same-team overlap counts as a piece),
  * missed possessions: duration, and whether a restart tag window covers
    them (``SET PIECES``-adjacent misses),
  * spurious possessions: duration of predicted spells matching nothing,
  * dead-ball run matching at IoU 0.3 (GT D-runs vs predicted D-runs).

Tune and held-out matches are aggregated separately; the held-out numbers
are the ones to quote (they played no part in selection).

Usage::

    python train/error_analysis.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402
from train.lab import list_matches, load_lab_match, pm, run_pipeline  # noqa: E402

FPS = 25
OUT_PATH = os.path.join(REPO_ROOT, "train", "error_analysis_results.json")

# First-principles values the sweep started from (same record the held-out
# check uses, train/validate_heldout.py).
FIRST_PRINCIPLES = dict(
    sigma_m=3.5, th_switch=0.25, th_motion_low=0.8, th_motion_high=1.7,
    th_ball_move=4.0, t_exit_frames=4, t_min_state_frames=15,
)


def _pred_flags(overlaps, n_pred, threshold):
    """Greedy one-to-one match flags on the predicted side (mirror of
    ``pm.matched_gt_flags``)."""
    flags = np.zeros(n_pred, dtype=bool)
    seen_gt = set()
    for iou, gi, pi in overlaps:
        if iou < threshold:
            break
        if flags[pi] or gi in seen_gt:
            continue
        flags[pi] = True
        seen_gt.add(gi)
    return flags


def _d_segments(codes, runs):
    return [(a, b, k) for a, b, k in pm.segments(codes, runs) if k == 2]


def analyse_match(match_id: str, cfg: Config) -> dict:
    tr, gt, tags, _ = load_lab_match(match_id)
    pred = run_pipeline(tr, tags, cfg)
    runs = pm.valid_runs(gt)

    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(pred, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    scores = pm.spell_scores(gt, pred, runs)
    flags03 = pm.matched_gt_flags(overlaps, len(gt_spells), 0.3)
    pflags03 = _pred_flags(overlaps, len(pred_spells), 0.3)

    # pieces: any same-team predicted spell touching a GT spell
    pieces = np.zeros(len(gt_spells), dtype=int)
    for _iou, gi, _pi in overlaps:
        pieces[gi] += 1

    def dur(a, b):
        return (b - a) / FPS

    missed = [gt_spells[i] for i in range(len(gt_spells)) if not flags03[i]]
    matched = [gt_spells[i] for i in range(len(gt_spells)) if flags03[i]]
    spurious = [pred_spells[j] for j in range(len(pred_spells)) if not pflags03[j]]

    tag_spans = [(t.start_f, t.end_f) for t in tags]
    def touches_tag(spell):
        a, b, _ = spell
        return any(b > ta and a < tb for ta, tb in tag_spans)

    # dead-ball runs, same greedy matching machinery as possessions
    d_gt = _d_segments(gt, runs)
    d_pred = _d_segments(pred, runs)
    d_ov = pm.spell_overlaps(d_gt, d_pred)
    d_flags = pm.matched_gt_flags(d_ov, len(d_gt), 0.3)

    return {
        "spell": scores["mean_f1"],
        "recall03": scores["per_iou"][0.3]["recall"],
        "prec03": scores["per_iou"][0.3]["precision"],
        "recall05": scores["per_iou"][0.5]["recall"],
        "n_gt": len(gt_spells),
        "n_pred": len(pred_spells),
        "pieces_mean": float(pieces.mean()) if len(pieces) else 0.0,
        "split_gt": int((pieces >= 2).sum()),
        "missed": len(missed),
        "missed_dur_s": [dur(a, b) for a, b, _ in missed],
        "matched_dur_s": [dur(a, b) for a, b, _ in matched],
        "missed_on_tag": sum(1 for s in missed if touches_tag(s)),
        "spurious": len(spurious),
        "spurious_dur_s": [dur(a, b) for a, b, _ in spurious],
        "d_recall03": float(d_flags.mean()) if len(d_flags) else 0.0,
        "n_d_gt": len(d_gt),
        "n_d_pred": len(d_pred),
    }


def _agg(rows: list[dict]) -> dict:
    def flat(key):
        return [x for r in rows for x in r[key]]

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    n_gt = sum(r["n_gt"] for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    missed = sum(r["missed"] for r in rows)
    spurious = sum(r["spurious"] for r in rows)
    return {
        "n_matches": len(rows),
        "spell_mean": mean([r["spell"] for r in rows]),
        "recall03_micro": (n_gt - missed) / n_gt if n_gt else 0.0,
        "prec03_micro": (n_pred - spurious) / n_pred if n_pred else 0.0,
        "recall05_mean": mean([r["recall05"] for r in rows]),
        "pieces_mean": mean([r["pieces_mean"] for r in rows]),
        "split_gt_pct": 100.0 * sum(r["split_gt"] for r in rows) / n_gt if n_gt else 0.0,
        "missed": missed,
        "missed_dur_s_mean": mean(flat("missed_dur_s")),
        "matched_dur_s_mean": mean(flat("matched_dur_s")),
        "missed_on_tag_pct": 100.0 * sum(r["missed_on_tag"] for r in rows) / missed
        if missed else 0.0,
        "spurious": spurious,
        "spurious_dur_s_mean": mean(flat("spurious_dur_s")),
        "d_recall03_mean": mean([r["d_recall03"] for r in rows]),
        "n_d_gt": sum(r["n_d_gt"] for r in rows),
        "n_d_pred": sum(r["n_d_pred"] for r in rows),
    }


def main() -> None:
    with open(os.path.join(REPO_ROOT, "train", "sweep_results.json"),
              encoding="utf-8") as f:
        tune = set(json.load(f)["tune_matches"])

    cfg_frozen = Config()
    cfg_defaults = Config.from_dict({**cfg_frozen.to_dict(), **FIRST_PRINCIPLES})

    ids = [m for m in list_matches()]
    results = {"frozen": {}, "defaults": {}}
    for name, cfg in (("frozen", cfg_frozen), ("defaults", cfg_defaults)):
        rows_tune, rows_held = [], []
        for mid in ids:
            r = analyse_match(mid, cfg)
            (rows_tune if mid in tune else rows_held).append(r)
            print(f"  {name} {mid}: spell {r['spell']:.4f} recall@0.3 "
                  f"{r['recall03']:.3f} pieces {r['pieces_mean']:.2f} "
                  f"missed {r['missed']} spurious {r['spurious']}", flush=True)
        results[name]["tune"] = _agg(rows_tune)
        results[name]["heldout"] = _agg(rows_held)

    for split in ("heldout", "tune"):
        print(f"\n=== {split} ===")
        header = ("metric", "frozen", "defaults")
        print(f"{header[0]:<26}{header[1]:>10}{header[2]:>10}")
        keys = [
            ("spell (mean per match)", "spell_mean", "{:.4f}"),
            ("recall@0.3 (micro)", "recall03_micro", "{:.3f}"),
            ("precision@0.3 (micro)", "prec03_micro", "{:.3f}"),
            ("recall@0.5 (mean)", "recall05_mean", "{:.3f}"),
            ("pieces per GT spell", "pieces_mean", "{:.2f}"),
            ("GT spells split >=2", "split_gt_pct", "{:.1f}%"),
            ("missed GT spells", "missed", "{:d}"),
            ("missed dur (s)", "missed_dur_s_mean", "{:.1f}"),
            ("matched dur (s)", "matched_dur_s_mean", "{:.1f}"),
            ("misses under restart tag", "missed_on_tag_pct", "{:.1f}%"),
            ("spurious pred spells", "spurious", "{:d}"),
            ("spurious dur (s)", "spurious_dur_s_mean", "{:.1f}"),
            ("D-run recall@0.3", "d_recall03_mean", "{:.3f}"),
            ("D-runs gt/pred", "n_d_gt", "{:d}"),
        ]
        for label, key, fmt in keys:
            if key == "n_d_gt":
                vals = [f"{results[c][split]['n_d_gt']}/{results[c][split]['n_d_pred']}"
                        for c in ("frozen", "defaults")]
                print(f"{label:<26}{vals[0]:>10}{vals[1]:>10}")
            else:
                vals = [fmt.format(results[c][split][key])
                        for c in ("frozen", "defaults")]
                print(f"{label:<26}{vals[0]:>10}{vals[1]:>10}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
