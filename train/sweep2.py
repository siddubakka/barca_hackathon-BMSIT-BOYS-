"""Stage-2 sweep: push past stage-1's grid edges, on the same 6 tune matches.

Stage-1 (train/sweep.py) converged with four parameters sitting *on the edge*
of their grids -- th_switch at its lowest, th_motion_low at its lowest,
t_exit_frames / t_min_state_frames / th_ball_move at their highest -- so the
coordinate descent was limited by the grid, not by the optimum.  This stage
extends those grids and refines the interior around the frozen optimum.

Selection rules (lab-legal end to end; the competition match and the public
5-minute GT are never consulted):

  * objective  -- lexicographic (mean spellF1, mean macro-F1): the ranking
                  metric and its official tie-break;
  * dead-volume band (hard filter) -- candidates whose mean graded dead share
                  moves below -3.0pp or above +1.5pp versus the frozen config
                  are rejected.  Volume is pinned because lab's dead share
                  (~45%) is a construction of SkillCorner's phase gaps, not
                  the competition's operating point: the search may improve
                  *placement* (timing, teams, boundaries) but must not chase
                  a lab-specific volume artifact;
  * soft tie-break -- after (spell, macro), closeness of the predicted/GT
                  spell-count ratio to 1.0 (less fragmentation).

Usage:
    python train/sweep2.py [--passes 3] [--out train/sweep2_results.json]
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

# Extended / refined grids.  Values in parentheses comment what stage-1 did.
CANDIDATES: dict[str, list] = {
    # -- stage-1 hit the LOW edge of these: extend downward ---------------
    "th_switch": [0.05, 0.075, 0.10, 0.14],           # stage-1 best 0.12 = edge
    "th_motion_low": [0.2, 0.28, 0.35, 0.45],         # stage-1 best 0.4 = edge
    # -- stage-1 hit the HIGH edge of these: extend upward ---------------
    "t_exit_frames": [16, 22, 30],                    # stage-1 best 12 = edge
    "t_min_state_frames": [55, 75, 100],              # stage-1 best 40 = edge
    "th_ball_move": [14, 20, 30],                     # stage-1 best 10 = edge
    # -- interior refinement around the frozen optimum -------------------
    "t_switch_frames": [20, 24, 36, 45],              # frozen 30 (default)
    "t_min_poss_frames": [60, 90, 110],               # frozen 75
    "sigma_m": [1.6, 1.9, 2.2, 3.0],                  # frozen 2.5 = stage-1 low edge
    "r_max_m": [24.0, 38.0],                          # frozen 30 (flat in stage-1)
    "max_hole_frames": [35, 65, 100],                 # frozen 50
    "th_motion_high": [1.65, 2.0, 2.1],               # frozen 1.8 (interior)
    "t_enter_frames": [4, 7],                         # frozen 5 (interior)
    # -- restart takeover (new in stage-2; default 30 = stage-1 behaviour) --
    "t_restart_flip_frames": [6, 10, 16, 22],         # frozen 30 (no-op)
}

D_VOL_DOWN = -0.030   # never hollow out dead volume beyond -3.0pp on lab
D_VOL_UP = +0.015     # never inflate dead volume beyond +1.5pp on lab


def evaluate(cfg: Config, matches: list[str], cache: dict, lab: dict) -> dict:
    key = json.dumps(cfg.to_dict(), sort_keys=True)
    if key in cache:
        return cache[key]
    t0 = time.time()
    rows, dshares = [], []
    for m in matches:
        tr, gt, tags, _stats = lab[m]
        pred = run_pipeline(tr, tags, cfg)
        rows.append(pm.evaluate(gt, pred))
        graded = gt != 3
        dshares.append(float((pred[graded] == 2).mean()) if graded.any() else 0.0)
    out = {
        "spell": sum(r["spell_f1_mean"] for r in rows) / len(rows),
        "macro": sum(r["macro_f1"] for r in rows) / len(rows),
        "alive_dead": sum(r["alive_dead_f1"] for r in rows) / len(rows),
        "spell_ratio": sum(
            r["n_pred_spells"] / max(1, r["n_gt_spells"]) for r in rows
        ) / len(rows),
        "d_share": sum(dshares) / len(dshares),
        "per_match_spell": [round(r["spell_f1_mean"], 4) for r in rows],
        "seconds": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
    }
    cache[key] = out
    return out


def rank_key(res: dict) -> tuple:
    """Official order first (spell, macro), then our structure tie-break."""
    return (res["spell"], res["macro"], -abs(res["spell_ratio"] - 1.0))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "train", "sweep2_results.json"),
    )
    args = ap.parse_args(argv)

    all_ids = list_matches()
    with open(os.path.join(REPO_ROOT, "train", "sweep_results.json"),
              encoding="utf-8") as f:
        tune = set(json.load(f)["tune_matches"])
    tune_ids = [m for m in all_ids if m in tune]
    assert len(tune_ids) == 6, tune_ids
    print(f"tuning on {len(tune_ids)} matches: {', '.join(tune_ids)}", flush=True)

    print("loading lab matches...", flush=True)
    lab = {m: load_lab_match(m) for m in tune_ids}

    cache: dict = {}
    base_cfg = Config()  # the frozen stage-1 selection
    base = evaluate(base_cfg, tune_ids, cache, lab)
    d0 = base["d_share"]
    lo, hi = d0 + D_VOL_DOWN, d0 + D_VOL_UP
    print(
        f"frozen baseline: spell {base['spell']:.4f}  macro {base['macro']:.4f}"
        f"  a/d {base['alive_dead']:.3f}  ratio {base['spell_ratio']:.2f}"
        f"  d_share {d0:.4f}  (band [{lo:.4f}, {hi:.4f}])"
        f"  ({base['seconds']}s)",
        flush=True,
    )

    trials: list[dict] = [dict(base, step="baseline", accepted=True)]
    history: list[str] = []
    best, best_res = base_cfg, base

    for p in range(args.passes):
        improved_any = False
        for param, values in CANDIDATES.items():
            local_best, local_res = best, best_res
            for v in values:
                if v == getattr(best, param):
                    continue
                cand = dataclasses.replace(best, **{param: v})
                res = evaluate(cand, tune_ids, cache, lab)
                accepted = lo <= res["d_share"] <= hi
                trials.append(
                    dict(res, step=f"pass{p}:{param}={v}", accepted=accepted)
                )
                marker = ""
                if not accepted:
                    marker = " (D-band reject)"
                elif rank_key(res) > rank_key(local_res):
                    local_best, local_res = cand, res
                    marker = " *"
                print(
                    f"  pass{p} {param}={v:<6} spell {res['spell']:.4f}"
                    f"  macro {res['macro']:.4f}  ratio {res['spell_ratio']:.2f}"
                    f"  d {res['d_share']:.4f}  ({res['seconds']}s){marker}",
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
            f"pass {p} done: spell {best_res['spell']:.4f}"
            f" macro {best_res['macro']:.4f} d {best_res['d_share']:.4f}",
            flush=True,
        )
        if not improved_any:
            break

    print("\nBEST CONFIG (stage-2)")
    print(json.dumps(best.to_dict(), indent=2))
    print("history:", " | ".join(history))
    print(
        f"tune: spell {best_res['spell']:.4f} (frozen {base['spell']:.4f}, "
        f"delta {best_res['spell'] - base['spell']:+.4f})  "
        f"macro {best_res['macro']:.4f} (frozen {base['macro']:.4f})  "
        f"per-match {best_res['per_match_spell']}"
    )
    wins = sum(
        a > b for a, b in zip(best_res["per_match_spell"], base["per_match_spell"])
    )
    print(f"per-match wins vs frozen: {wins}/6")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tune_matches": tune_ids,
                "baseline_frozen": {
                    k: base[k] for k in
                    ("spell", "macro", "alive_dead", "spell_ratio", "d_share",
                     "per_match_spell")
                },
                "d_band": [lo, hi],
                "best": best.to_dict(),
                "best_metrics": {
                    k: best_res[k] for k in
                    ("spell", "macro", "alive_dead", "spell_ratio", "d_share",
                     "per_match_spell")
                },
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
