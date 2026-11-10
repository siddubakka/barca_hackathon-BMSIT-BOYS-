"""Window-vs-full-match compression stress test on legal lab data.

Why this exists
---------------
The competition's only public signal is the first 5 minutes of graded play;
the ranking is computed on the full match.  The official baselines compress
between those two views of the *same* prediction::

    baseline_smooth50   window 0.3256 -> full 0.2433   (delta -0.082)
    baseline_smooth125  window 0.4242 -> full 0.2242   (delta -0.200)

so a good window number does not imply a good ranking number (EVALUATION.md
calls smooth125 the trap).  The public window may be checked only once and
may never guide changes -- but measuring OUR OWN window-vs-full delta on the
SkillCorner lab is fully legal: decode each lab match once, then score the
identical prediction twice, whole graded span versus 5-minute slices,
exactly the way the scorer restricts itself to the public window.

How to read it
--------------
    delta = spellF1(5-minute slice) - spellF1(full match)

The opening slice of a block mirrors the competition window most closely:
the decode starts cold at the block boundary, exactly as the competition
decode starts cold at graded frame 11185.  Delta near zero means the window
check we passed is an unbiased preview of the full-match score; a strongly
negative delta would mean windows systematically overstate us.

Usage::

    python train/window_stress.py
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

WIN_S = 300  # the public window is the first 5 minutes of graded play
WIN_F = WIN_S * 25  # lab grid is 25 fps, same as the competition
OUT_PATH = os.path.join(REPO_ROOT, "train", "window_stress_results.json")


def _score(gt: np.ndarray, pred: np.ndarray) -> dict:
    r = pm.evaluate(gt, pred)
    return {
        "spell": float(r["spell_f1_mean"]),
        "macro": float(r["macro_f1"]),
        "gt_spells": int(r.get("n_gt_spells", -1)),
        "pred_spells": int(r.get("n_pred_spells", -1)),
    }


def _fmt(x: np.ndarray) -> str:
    if x.size == 0:
        return "n/a"
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return (f"mean {x.mean():+.4f}  sd {sd:.4f}  "
            f"min {x.min():+.4f}  max {x.max():+.4f}  n {x.size}")


def main() -> None:
    cfg = Config()
    ids = list_matches()
    print(f"{len(ids)} lab matches")
    rows: list[dict] = []
    for mid in ids:
        tr, gt, tags, _stats = load_lab_match(mid)
        pred = run_pipeline(tr, tags, cfg)
        full = _score(gt, pred)
        wins = []
        for bi, (a, b) in enumerate(tr.meta.get("blocks") or ((0, tr.n),)):
            if b - a < WIN_F:
                continue
            wins.append((f"open{bi + 1}", a, a + WIN_F, bi == 0))
            wins.append((f"close{bi + 1}", b - WIN_F, b, False))
        parts = []
        for name, s, e, is_open in wins:
            w = _score(gt[s:e], pred[s:e])
            rows.append({
                "match": str(mid),
                "window": name,
                "is_match_open": bool(is_open),
                "full_spell": full["spell"],
                "full_macro": full["macro"],
                "win_spell": w["spell"],
                "win_macro": w["macro"],
                "win_gt_spells": w["gt_spells"],
                "d_spell": w["spell"] - full["spell"],
                "d_macro": w["macro"] - full["macro"],
            })
            parts.append(f"{name} {w['spell']:.4f} "
                         f"({w['spell'] - full['spell']:+.4f})")
        tail = "  ".join(parts) if parts else "(no block >= 5 min)"
        print(f"  {mid}: full {full['spell']:.4f}/{full['macro']:.4f} | {tail}",
              flush=True)

    if not rows:
        print("no windows scored (no lab data?)")
        return

    d_all = np.array([r["d_spell"] for r in rows], dtype=float)
    d_open = np.array([r["d_spell"] for r in rows if r["is_match_open"]],
                      dtype=float)
    m_all = np.array([r["d_macro"] for r in rows], dtype=float)
    m_open = np.array([r["d_macro"] for r in rows if r["is_match_open"]],
                      dtype=float)

    print("\nwindow - full delta, spellF1")
    print(f"  all slices        : {_fmt(d_all)}")
    print(f"  match-open slices : {_fmt(d_open)}   (competition analog)")
    print("window - full delta, macroF1 (first tie-break)")
    print(f"  all slices        : {_fmt(m_all)}")
    print(f"  match-open slices : {_fmt(m_open)}")
    print("\nreference -- official baselines on the competition match:")
    print("  smooth50  -0.082    smooth125  -0.200")

    out = {
        "n_matches": len(ids),
        "window_frames": int(WIN_F),
        "d_spell_all_mean": float(d_all.mean()),
        "d_spell_open_mean": float(d_open.mean()) if d_open.size else None,
        "d_spell_all_sd": float(np.std(d_all, ddof=1)) if d_all.size > 1 else None,
        "d_macro_open_mean": float(m_open.mean()) if m_open.size else None,
        "rows": rows,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
