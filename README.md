# mtah2027-possession

Frame-level ball-possession decoding for **More than a Hack 2027 — Phase 1:
Ball Possession** (Metrica Sports / FC Barcelona), tactical source `BAR-ATH`.
Team: **BMSITBOYS**.

Given the per-frame tracking bundle (ATD raw text + metadata XML, homography,
Smart Tagging XML), the pipeline emits one `H / A / D / X` label per video
frame for all 199,319 frames of the match.

Full method write-up: **[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)**
(§7 = measurements, §8 = limitations and fixes, §9 = reproduction).

## Results

Public 5-minute pipeline check (used once, never for tuning):
**spellF1 0.3545 — 2nd of 5 systems; macroF1 0.6627 — 1st**, both above
`smooth50` (0.3256 / 0.6207 on the same window) — the baseline that wins
the full match. The challenge's own caveat applies: with 17 ground-truth
possessions in the window one miss moves spellF1 ≈ 0.06, so these gaps are
indicative; the generalization evidence is the **+0.078** held-out gain
over first-principles defaults on 14 unseen lab matches (report §7).

The submitted file is byte-reproducible from this repository
(`python make_submission.py` below), SHA256
`8CEE6FE3A39F2715F993DA595E66351BB476EDD8B302CB06A6DB262292188E99`.

## Reproduce the submission

```
pip install -r requirements.txt
python make_submission.py --data-dir /path/to/data --out submission.csv
```

`--data-dir` (or the `MTAH_DATA_DIR` environment variable) must point at a
directory containing the competition assets, either directly or under a
`tactical/` subdirectory (Windows and POSIX paths both work):

```
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_homography.csv
BAR-ATH_tactical_smart_tagging.xml
```

The run is deterministic: same data + config => byte-identical CSV
(~11 s with the parse cache warm, ~60 s cold). Format is validated
automatically before the file is written (`frame,label`, 199,319 rows,
frame 0..N-1 in order).

Validate independently against the challenge's own checker (the second
command runs from inside the challenge clone):

```
python ../more-than-a-hack-2027/evaluation/validate_submission.py submission.csv
python evaluation/run_evaluation.py          # in the challenge repo
```

Run the unit tests (no extra dependencies):

```
python tests/test_core.py
```

## How it works

```
src/load.py      parse ATD raw tracking, metadata XML, homography, Smart Tagging
src/repair.py    interpolate short ball holes, geometry ball speed, player motion
src/evidence.py  Gaussian proximity team vote vs the ball + Smart Tagging priors
src/decode.py    two state machines: dead/alive (hysteresis), possession
                 (sustained-evidence switching, hold-through-lost-ball)
src/emit.py      CSV writer + format validator
```

Design rules that come from the competition brief:

* **Ball missing != dead.** The baseline treats "no ball vote in a ±50-frame
  window" as `D`; here a lost ball freezes the team state (possession stays
  with the last team during loose phases), while `D` requires independent
  dead-ball evidence: restart windows from Smart Tagging, the ball being
  tracked outside the touchline, a median-player-speed lull, or a
  kicked-ball/running-players exit.
* **Switches are controlled.** A team change needs `t_switch_frames` of
  sustained opposite evidence and never fires before `t_min_poss_frames` of
  dwell; a restart resets the dwell so the team taking the restart can take
  over promptly.
* **Published frame ranges are hard anchors**: frames 110808–113741 (pre
  second-half kickoff) are emitted `D`, and all ungraded stretches (pre-match,
  halftime, post-match) are `D` as well.

## Tuning protocol

All hyperparameters (`src/config.py`) were selected on a **SkillCorner
open-data practice lab** (`train/lab.py`, `train/sweep.py`): 6 labeled matches
resampled to the competition's 25 fps grid, in-play ball visibility degraded
to the competition's estimated level, scored with the challenge's own
`possession_metrics.evaluate` by mean `spell_f1_mean`. The competition match
and the public 5-minute ground truth were never used to pick a configuration;
the public window is used once as a pipeline check. A window-vs-full
compression stress (`train/window_stress.py`) confirms the window check is a
fair preview: our 5-minute slices land within 0.03 of the full-match score
where the official baselines lose 0.08–0.20 between the same two views.

```
python train/sweep.py --passes 2 --matches 6     # ~1 h, re-runs the selection
                                                 # (writes train/sweep_results.json)
```

The lab scripts (`train/*.py`) locate their data through the environment —
`SKILLCORNER_DIR` (the SkillCorner match files) and `LAB_CACHE_DIR` (parse
cache); without them they report 0 matches and exit cleanly. They are **not**
needed to reproduce the submission; only `make_submission.py` and
`tests/test_core.py` are.

## Repository layout

```
make_submission.py     single-command entry point
src/                   pipeline modules (load/repair/evidence/decode/emit/config/constants)
train/lab.py           SkillCorner practice lab (GT construction, degradation, scoring)
train/sweep.py         coordinate-descent hyperparameter sweep on the lab
train/validate_heldout.py  held-out generalization check (14 lab matches)
train/window_stress.py    window-vs-full compression stress (legal lab check)
train/error_analysis.py   failure-mode decomposition (misses/splits/spurious)
tests/test_core.py     unit tests (python tests/test_core.py)
docs/TECHNICAL_REPORT.md   approach, assets used/dropped, training data,
                           possession rules, limitations, reproduction
```

Match data is **not** committed to this repository (competition rule); the
code regenerates the submission from the downloaded data bundle alone.

## License

MIT — see [LICENSE](LICENSE).
