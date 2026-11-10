"""Tunable configuration for the possession decoder.

Every number here is a hyperparameter.  Selection protocol (per playbook):

  * defaults are set from first principles / signal inspection (no labels),
  * the chosen values are tuned on the SkillCorner practice lab only,
  * the public 5-minute GT is used once at the end as a pipeline check,
    never to pick between variants.

The values below are the FROZEN selection of ``train/sweep.py`` (coordinate
descent, mean spellF1 over 6 tune matches, macro tie-break; tune-set means
spell 0.3683 / macro 0.6145 vs 0.2969 / 0.6714 for the first-principles
defaults).  The full trial log lives in ``train/sweep_results.json``; each
field comment records what the sweep chose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- ball repair ----------------------------------------------------
    # linear interpolation is only trusted across short holes; longer holes
    # leave the ball unknown (the evidence module then holds the last team).
    max_hole_frames: int = 50          # 2 s at 25 fps -- sweep kept the default

    # ---- team evidence (tracking) ---------------------------------------
    sigma_m: float = 2.5               # sweep: 2.5 (grid 2.5/3.5/5/7)
    r_max_m: float = 30.0              # sweep kept the default (20/30/45 tie)

    # ---- weak team evidence from Smart Tagging ---------------------------
    # Only codes whose semantics are unambiguous are used; see evidence.py.
    w_tag: float = 0.30                # not tunable in the lab (no phase tags)

    # ---- possession state machine ----------------------------------------
    th_switch: float = 0.12            # sweep: 0.12 (grid .12/.18/.25/.35/.5)
    t_switch_frames: int = 30          # sweep kept the default (12/25/40/60)
    t_min_poss_frames: int = 75        # sweep kept the default (25/50/75/125/200)

    # ---- dead / alive state machine --------------------------------------
    th_motion_low: float = 0.4         # sweep: 0.4 (grid .4/.6/.8/1.0/1.3)
    th_motion_high: float = 1.8        # sweep: 1.8 (grid 1.2/1.5/1.8/2.2)
    th_ball_move: float = 10.0         # sweep: 10.0 (grid 4/7/10)
    t_enter_frames: int = 5            # sweep kept the default (3/5/10)
    t_exit_frames: int = 12            # sweep: 12 (grid 3/4/8/12)
    t_min_state_frames: int = 40       # sweep: 40 (grid 8/15/25/40)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)
