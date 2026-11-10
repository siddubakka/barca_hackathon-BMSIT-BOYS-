#!/usr/bin/env python
"""Generate the possession submission CSV from the competition data bundle.

Usage:
    python make_submission.py --data-dir C:\\path\\to\\data [--out submission.csv]

``--data-dir`` defaults to the MTAH_DATA_DIR environment variable.  The data
directory must contain ``tactical/BAR-ATH_tactical_*`` assets (or the files
directly).  Deterministic: the same data and config always produce the same
file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import load
from src.config import Config
from src.constants import GRADED_BLOCKS, N_VIDEO_FRAMES
from src.decode import decode
from src.emit import labels_to_csv, validate
from src.evidence import compute
from src.repair import repair

LABEL_STR = np.array(["H", "A", "D", "X"])


def build_labels(data_dir: str, cfg: Config, quiet: bool = False) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    paths = load.asset_paths(data_dir)
    tr = load.load_atd(paths["atd_meta"], paths["atd_raw"])
    tags = load.load_smart_tagging(paths["smart_tagging"])
    rep = repair(tr, cfg)
    ev = compute(tr, rep, tags, cfg)
    labels = decode(tr, rep, ev, cfg, n=N_VIDEO_FRAMES)
    stats = {
        "parse_s": round(time.time() - t0, 1),
        "n_tags": len(tags),
        "ball_valid_pct": round(100 * float(rep.ball_valid.mean()), 1),
        "motion_median": round(float(np.median(rep.motion)), 2),
    }
    if not quiet:
        counts = np.bincount(labels, minlength=4)
        stats["label_counts"] = {
            "H": int(counts[0]), "A": int(counts[1]), "D": int(counts[2]), "X": int(counts[3]),
        }
    return labels, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("MTAH_DATA_DIR"),
        help="directory with the BAR-ATH tactical assets (or MTAH_DATA_DIR)",
    )
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "submission.csv"))
    ap.add_argument("--config", default=None, help="JSON file with Config overrides")
    args = ap.parse_args(argv)
    if not args.data_dir:
        ap.error("--data-dir or the MTAH_DATA_DIR environment variable is required")

    cfg = Config()
    if args.config:
        import json

        with open(args.config, encoding="utf-8") as f:
            cfg = Config.from_dict(json.load(f))

    labels, stats = build_labels(args.data_dir, cfg)
    labels_to_csv(labels, args.out, n_expected=N_VIDEO_FRAMES)
    validate(args.out, N_VIDEO_FRAMES)

    # segment structure of our own output (no ground truth involved)
    graded = np.zeros(N_VIDEO_FRAMES, dtype=bool)
    for a, b in GRADED_BLOCKS:
        graded[a:b] = True
    segs = []
    start = None
    for i in range(N_VIDEO_FRAMES):
        if not graded[i]:
            if start is not None:
                segs.append((start, i, labels[start]))
                start = None
            continue
        if start is None:
            start = i
        elif labels[i] != labels[start]:
            segs.append((start, i, labels[start]))
            start = i
    if start is not None:
        segs.append((start, N_VIDEO_FRAMES, labels[start]))
    poss = [s for s in segs if s[2] in (0, 1)]
    dsegs = [s for s in segs if s[2] == 2]
    play_min = graded.sum() / 25 / 60
    d_share = 100 * int(((labels == 2) & graded).sum()) / max(1, int(graded.sum()))

    print(f"wrote {args.out}")
    print(f"  rows: {N_VIDEO_FRAMES}  parse+decode: {stats['parse_s']}s")
    print(f"  ball valid: {stats['ball_valid_pct']}%  tags: {stats['n_tags']}")
    print(f"  labels: {stats['label_counts']}")
    print(
        f"  structure (graded): {len(poss)} possessions ({len(poss)/play_min:.1f}/min),"
        f" {len(dsegs)} D runs ({len(dsegs)/play_min:.1f}/min),"
        f" D share {d_share:.1f}%"
    )
    print("  local format validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
