"""Coordinate-descent sweep of src/config.py on the SkillCorner practice lab.

Selection rule (playbook): pick by MEAN spellF1 over the tune matches -- the
ranking metric of the competition -- with mean macro-F1 as tie-break.  The
competition match and the public 5-minute GT are never consulted here.

Usage:
    python train/sweep.py [--passes 2] [--matches 8] [--out train/sweep_results.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402
from train.lab import list_matches, load_lab_match, pm, run_pipeline  # noqa: E402

# Candidate grids.  Values bracket the first-principles defaults; the lab
# decides.  Frames are at 25 fps (40 ms), so t_*_frames = seconds * 25.
CANDIDATES: dict[str, list] = {
    # dead / alive machine
    "th_motion_low": [0.4, 0.6, 0.8, 1.0, 1.3],
    "th_motion_high": [1.2, 1.5, 1.8, 2.2],
    "th_ball_move": [4.0, 7.0, 10.0],
    "t_enter_frames": [3, 5, 10],
    "t_exit_frames": [3, 4, 8, 12],
    "t_min_state_frames": [8, 15, 25, 40],
    # possession machine
    "th_switch": [0.12, 0.18, 0.25, 0.35, 0.5],
    "t_switch_frames": [12, 25, 40, 60],
    "t_min_poss_frames": [25, 50, 75, 125, 200],
    # team evidence
    "sigma_m": [2.5, 3.5, 5.0, 7.0],
    "r_max_m": [20.0, 30.0, 45.0],
    # ball repair
    "max_hole_frames": [12, 25, 50, 100, 250],
}


def evaluate(cfg: Config, matches: list[str], cache: dict, lab: dict) -> dict:
    key = json.dumps(cfg.to_dict(), sort_keys=True)
    if key in cache:
        return cache[key]
    t0 = time.time()
    rows = []
    for m in matches:
        tr, gt, tags, _stats = lab[m]
        pred = run_pipeline(tr, tags, cfg)
        rows.append(pm.evaluate(gt, pred))
    out = {
        "spell": sum(r["spell_f1_mean"] for r in rows) / len(rows),
        "macro": sum(r["macro_f1"] for r in rows) / len(rows),
        "alive_dead": sum(r["alive_dead_f1"] for r in rows) / len(rows),
        "spell_ratio": sum(
            r["n_pred_spells"] / max(1, r["n_gt_spells"]) for r in rows
        )
        / len(rows),
        "per_match_spell": [round(r["spell_f1_mean"], 4) for r in rows],
        "seconds": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
    }
    cache[key] = out
    return out


def better(a: dict, b: dict) -> bool:
    """Lexicographic: mean spellF1 first (the ranking metric), macro tie-break."""
    return (a["spell"], a["macro"]) > (b["spell"], b["macro"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "train", "sweep_results.json"))
    args = ap.parse_args(argv)

    all_ids = list_matches()
    tune = all_ids[: args.matches]
    print(f"tuning on {len(tune)} matches: {', '.join(tune)}", flush=True)

    print("loading lab matches...", flush=True)
    lab = {m: load_lab_match(m) for m in tune}
    for m in tune:
        print(f"  {m}: {lab[m][3]}", flush=True)

    cache: dict = {}
    best = Config()
    best_res = evaluate(best, tune, cache, lab)
    print(
        f"baseline: spell {best_res['spell']:.4f}  macro {best_res['macro']:.4f}"
        f"  alive_dead {best_res['alive_dead']:.3f}"
        f"  pred/gt spells {best_res['spell_ratio']:.2f}"
        f"  ({best_res['seconds']}s)",
        flush=True,
    )

    trials: list[dict] = [dict(best_res, step="baseline")]
    history: list[str] = []

    for p in range(args.passes):
        improved_any = False
        for param, values in CANDIDATES.items():
            local_best, local_res = best, best_res
            for v in values:
                if v == getattr(best, param):
                    continue
                cand = dataclasses.replace(best, **{param: v})
                res = evaluate(cand, tune, cache, lab)
                trials.append(dict(res, step=f"pass{p}:{param}={v}"))
                marker = ""
                if better(res, local_res):
                    local_best, local_res = cand, res
                    marker = " *"
                print(
                    f"  pass{p} {param}={v:<6} spell {res['spell']:.4f}"
                    f"  macro {res['macro']:.4f}  ratio {res['spell_ratio']:.2f}"
                    f"  ({res['seconds']}s){marker}",
                    flush=True,
                )
            if local_best is not best:
                history.append(f"pass{p}: {param} -> {getattr(local_best, param)}")
                best, best_res = local_best, local_res
                improved_any = True
                print(
                    f"  >> {param} = {getattr(best, param)}  "
                    f"(spell {best_res['spell']:.4f} macro {best_res['macro']:.4f})",
                    flush=True,
                )
        print(
            f"pass {p} done: spell {best_res['spell']:.4f} macro {best_res['macro']:.4f}",
            flush=True,
        )
        if not improved_any:
            break

    print("\nBEST CONFIG")
    print(json.dumps(best.to_dict(), indent=2))
    print("history:", " | ".join(history))
    print(f"per-match spellF1: {best_res['per_match_spell']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tune_matches": tune,
                "best": best.to_dict(),
                "best_metrics": {k: best_res[k] for k in
                                 ("spell", "macro", "alive_dead", "spell_ratio",
                                  "per_match_spell")},
                "history": history,
                "trials": trials,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
