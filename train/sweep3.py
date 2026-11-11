"""Stage-3 targeted trials + pre-registered final selection (2026-11-11).

PRE-REGISTRATION -- written BEFORE any stage-3 trial was evaluated and before
any candidate has been looked at on held-out data.

(A) SELECTION RULE v2 -- one deterministic pass over the pool of all recorded
    tune trials (stage-2 `sweep2_results.json` trials + the stage-3 trials run
    by this script; the stage-1 maximum equals the frozen config, 0.3683, which
    is below the eligibility threshold and cannot qualify):

    1. hard filter -- tune graded dead share inside the stage-2 band
       [frozen - 3.0pp, frozen + 1.5pp] (volume guard; lab's dead share is a
       phase-gap artefact, see report 8.1, so volume must not be chased).
    2. S* = max mean tune spellF1 over the pool; E = {spell >= S* - EPS},
       EPS = 0.004.  Rationale: on the 6 tune matches the paired standard
       error between near-top candidates observed in stage-2 ranges
       0.0001..0.0024 (largest 0.0024); EPS ~= 2x that, i.e. spell
       differences inside EPS cannot be resolved on the selection data.
    3. within E pick max mean tune macroF1 -- the official first tie-break,
       applied at the ranking metric's noise floor rather than at exact
       equality (unattainable on 6 noisy matches).  Ties: |spell_ratio - 1|,
       then higher spell.
    4. the winner is the SINGLE candidate evaluated on held-out (once).

    No other information may influence this pick -- no competition-window
    data (forbidden), no narrative preference.

(B) ADOPTION BAR (as pre-registered at stage-2, unchanged): adopt the
    selected config only if the one held-out run over the 14 lab matches
    shows mean spellF1 >= 0.3487 + 0.010, per-match wins >= 9/14 against the
    frozen config evaluated in the same run, tune dead share in band (A.1),
    held-out macroF1 >= 0.5786 - 0.010 (no material macro regression), and a
    documented mechanism.  Otherwise keep the frozen configuration and record
    stage-3 as explored-and-rejected.

Fixed trial list (no adaptivity -- every candidate below is evaluated exactly
once, from the stage-2 best config unless noted):

  EDGES      untried directions at stage-1/stage-2 grid edges:
             th_motion_low 0.15 0.10 0.50 0.55 (0.20 and 0.45 were edges),
             th_switch 0.04 0.03 (0.05 was the edge), sigma_m 3.4 3.8 (3.0
             the edge), max_hole 30 25 (35 the edge), th_motion_high 2.2,
             t_min_state 48, t_switch 40, th_ball_move 8, w_tag 0.15 0.5 0.7
             (never tuned in stage-1 or stage-2).
  FRONTIER   denser th_motion_low samples (0.24 0.32 0.40) to resolve the
             spell/macro trade surface the selection rule walks on.
  PAIRS      joint two-parameter moves coordinate descent never tried.
  SANITY     re-evaluation of the stage-2 best; must reproduce exactly or the
             environment has drifted and the run aborts.

    python train/sweep3.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import Config  # noqa: E402
from train.lab import list_matches, load_lab_match  # noqa: E402
from train.sweep2 import evaluate  # noqa: E402

EPS = 0.004          # rule A.2 -- eligibility margin on tune spellF1
HELD_BASE = 0.3487   # frozen held-out spellF1 (baseline for the adoption bar)
HELD_GAIN = 0.010    # minimum mean held-out gain to adopt
HELD_WINS = 9        # minimum per-match wins of 14 to adopt
HELD_MACRO = 0.5786  # frozen held-out macroF1
MACRO_TOL = 0.010    # allowed held-out macro regression

EDGES: dict[str, list] = {
    "th_motion_low": [0.15, 0.10, 0.50, 0.55],
    "th_switch": [0.04, 0.03],
    "sigma_m": [3.4, 3.8],
    "max_hole_frames": [30, 25],
    "th_motion_high": [2.2],
    "t_min_state_frames": [48],
    "t_switch_frames": [40],
    "th_ball_move": [8.0],
    "w_tag": [0.15, 0.5, 0.7],
}

FRONTIER: dict[str, list] = {"th_motion_low": [0.24, 0.32, 0.40]}

PAIRS: list[dict] = [
    {"th_motion_low": 0.28, "th_switch": 0.04},
    {"th_motion_low": 0.45, "th_switch": 0.04},
    {"th_motion_low": 0.28, "sigma_m": 3.4},
    {"th_motion_low": 0.45, "sigma_m": 3.4},
    {"th_motion_low": 0.28, "th_motion_high": 2.1},
    {"th_motion_low": 0.45, "th_motion_high": 2.1},
    {"th_motion_low": 0.28, "t_min_state_frames": 48},
    {"th_motion_low": 0.45, "t_min_state_frames": 48},
    {"th_switch": 0.04, "t_switch_frames": 40},
    {"th_switch": 0.04, "t_min_state_frames": 48},
    {"sigma_m": 3.4, "th_motion_high": 2.1},
    {"max_hole_frames": 30, "th_motion_low": 0.28},
]


def build_trials(base: Config) -> list[tuple[str, dict]]:
    """Fixed list of (step, overrides) -- EDGES, FRONTIER, then PAIRS."""
    out: list[tuple[str, dict]] = []
    for block in (EDGES, FRONTIER):
        for param, values in block.items():
            for v in values:
                if v != getattr(base, param):
                    out.append((f"{param}={v}", {param: v}))
    for i, over in enumerate(PAIRS):
        out.append((f"pair{i:02d}:" + "+".join(f"{k}={v}" for k, v in over.items()),
                    over))
    # dedupe identical effective configs (keep first)
    seen, uniq = set(), []
    for step, over in out:
        key = json.dumps({**base.to_dict(), **over}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append((step, over))
    return uniq


def main(argv: list[str] | None = None) -> int:
    out_path = os.path.join(REPO_ROOT, "train", "sweep3_results.json")
    if argv:
        out_path = argv[0] if not argv[0].startswith("-") else out_path

    with open(os.path.join(REPO_ROOT, "train", "sweep2_results.json"),
              encoding="utf-8") as f:
        s2 = json.load(f)
    band = s2["d_band"]
    pool: list[dict] = list(s2["trials"])          # stage-2 pool, configs attached
    best2 = Config.from_dict(s2["best"])
    best2_metrics = s2["best_metrics"]

    all_ids = list_matches()
    tune = set(s2["tune_matches"])
    tune_ids = [m for m in all_ids if m in tune]
    assert len(tune_ids) == 6, tune_ids
    print("PRE-REGISTERED SELECTION RULE v2 (stage-3 docstring): "
          f"band {band}, EPS {EPS}, max-macro within EPS, single held-out run",
          flush=True)
    print(f"tuning on {len(tune_ids)} matches: {', '.join(tune_ids)}", flush=True)

    print("loading lab matches...", flush=True)
    lab = {m: load_lab_match(m) for m in tune_ids}
    cache: dict = {}

    # -- SANITY: the stage-2 best must reproduce exactly -------------------
    t0 = time.time()
    rerun = evaluate(best2, tune_ids, cache, lab)
    ok = (abs(rerun["spell"] - best2_metrics["spell"]) < 1e-9
          and abs(rerun["macro"] - best2_metrics["macro"]) < 1e-9
          and abs(rerun["d_share"] - best2_metrics["d_share"]) < 1e-9)
    print(f"sanity rerun of stage-2 best: spell {rerun['spell']:.6f} "
          f"macro {rerun['macro']:.6f} d {rerun['d_share']:.6f} "
          f"({rerun['seconds']}s) exact={ok}", flush=True)
    if not ok:
        print("ABORT: environment drift -- stage-2 numbers do not reproduce",
              file=sys.stderr, flush=True)
        return 2

    trials: list[dict] = []
    plan = build_trials(best2)
    print(f"{len(plan)} fixed trials", flush=True)
    for i, (step, over) in enumerate(plan, 1):
        cfg = dataclasses.replace(best2, **over)
        res = evaluate(cfg, tune_ids, cache, lab)
        in_band = band[0] <= res["d_share"] <= band[1]
        trials.append(dict(res, step=step, in_band=in_band))
        print(f"[{i}/{len(plan)}] {step:<58} spell {res['spell']:.6f} "
              f"macro {res['macro']:.6f} d {res['d_share']:.4f} "
              f"{'band-ok' if in_band else 'BAND-REJECT'} "
              f"({res['seconds']}s)", flush=True)

    # -- rule A: one deterministic pass over stage-2 + stage-3 -------------
    stage3 = trials
    pool.extend(stage3)
    eligible_pool = [t for t in pool
                     if band[0] <= t["d_share"] <= band[1] and "config" in t]
    s_star = max(t["spell"] for t in eligible_pool)
    e_set = [t for t in eligible_pool if t["spell"] >= s_star - EPS]
    e_set.sort(key=lambda t: (t["macro"], -abs(t["spell_ratio"] - 1.0),
                              t["spell"]), reverse=True)
    selected = e_set[0]

    print(f"\nS* (max in-band tune spell) = {s_star:.6f}; "
          f"E-set = {len(e_set)} configs within {EPS}", flush=True)
    print("E-set ranked by rule (macro, |ratio-1|, spell):", flush=True)
    for t in e_set[:12]:
        mark = "  <== SELECTED" if t is selected else ""
        print(f"  spell {t['spell']:.6f} macro {t['macro']:.6f} "
              f"ratio {t['spell_ratio']:.3f} d {t['d_share']:.4f} "
              f"[{t.get('step', '?')}]{mark}", flush=True)
    print(f"\nSELECTED: {json.dumps(selected['config'], sort_keys=True)}",
          flush=True)
    print(f"  tune spell {selected['spell']:.6f} "
          f"(frozen {best2_metrics['spell']:.6f}, "
          f"delta {selected['spell'] - best2_metrics['spell']:+.4f})  "
          f"macro {selected['macro']:.6f} "
          f"(frozen {best2_metrics['macro']:.6f})  "
          f"d {selected['d_share']:.4f}", flush=True)
    print("next: the single held-out run "
          f"(adoption bar: >= {HELD_BASE + HELD_GAIN:.4f} mean, "
          f">= {HELD_WINS}/14 wins, macro >= {HELD_MACRO - MACRO_TOL:.4f})",
          flush=True)

    payload = {
        "preregistered": {
            "date": "2026-11-11",
            "rule": "A.1 band filter; A.2 spell >= S* - 0.004; "
                    "A.3 max macro, then |ratio-1|, then spell; "
                    "A.4 single held-out candidate",
            "bar": f"held-out mean >= {HELD_BASE + HELD_GAIN:.4f}, "
                   f"wins >= {HELD_WINS}/14, macro >= "
                   f"{HELD_MACRO - MACRO_TOL:.4f}, band, mechanism",
        },
        "band": band,
        "s_star": s_star,
        "sanity_rerun_exact": ok,
        "trials": trials,
        "eligible": [{k: t[k] for k in
                      ("step", "spell", "macro", "spell_ratio", "d_share",
                       "config")} for t in e_set],
        "selected": selected,
        "pool_size": len(pool),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {out_path} ({payload['elapsed_s']}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
