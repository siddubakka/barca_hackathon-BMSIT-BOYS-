"""Temporal decoder: evidence -> frame labels (H / A / D).

Two coupled state machines run over each active play block:

  dead/alive  -- hysteresis on median player speed, restart windows,
                 ball speed, and the ball being tracked outside the
                 touchline.  Entering dead needs sustained evidence
                 (restart windows and ball-out enter fast, motion lulls
                 at double that duration); leaving dead needs a kick,
                 running players, or the restart window closing -- and
                 is blocked while the ball is still outside the pitch.
                 Both states have a minimum dwell so the label cannot
                 flicker.

  possession  -- the team state (H/A) only updates while the ball is in
                 play.  A switch requires ``t_switch_frames`` frames of
                 fresh, sustained evidence against the current team (a
                 controlled turnover) and never fires before
                 ``t_min_poss_frames`` of dwell.  When the tracking vote
                 is missing (ball lost), evidence is frozen -- the state
                 holds, which implements the competition rule that
                 possession stays with the last team during loose phases.
                 A dead -> alive transition counts as satisfied dwell, so
                 the team taking the restart can take over after only
                 ``t_switch_frames`` of sustained evidence.

Frames outside the active blocks (pre-match, halftime, the published
pre-second-half-kickoff stretch, post-match) are emitted as D -- the ball
is not in play there; all of it except the published stretch is ungraded
anyway.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .constants import GRADED_BLOCKS, N_VIDEO_FRAMES, SECOND_HALF_PRE_KICKOFF
from .evidence import Evidence
from .load import Track
from .repair import Repaired

HOME, AWAY, DEAD = 0, 1, 2

# Blocks where play actually happens.  The second half block starts at the
# published kickoff (113742), not at the graded-block start (110808): those
# first frames are graded and hard-coded D per the FAQ.
ACTIVE_BLOCKS = (
    (GRADED_BLOCKS[0][0], GRADED_BLOCKS[0][1]),
    (SECOND_HALF_PRE_KICKOFF[1], GRADED_BLOCKS[1][1]),
)


def decode(
    tr: Track,
    rep: Repaired,
    ev: Evidence,
    cfg: Config,
    n: int = N_VIDEO_FRAMES,
    active_blocks: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """Label frames 0..n-1.  ``active_blocks`` defaults to the competition's
    graded play blocks; the practice lab passes ((0, n),) instead."""
    blocks = ACTIVE_BLOCKS if active_blocks is None else active_blocks
    labels = np.full(n, DEAD, dtype=np.int8)
    for a, b in blocks:
        a = max(0, a)
        b = min(n, b)
        if b > a:
            _decode_block(rep, ev, cfg, a, b, labels)
    return labels


def _decode_block(
    rep: Repaired, ev: Evidence, cfg: Config, a: int, b: int, out: np.ndarray
) -> None:
    m = b - a
    motion = rep.motion[a:b]
    motion_known = rep.motion_n[a:b] >= 3  # <3 tracked players -> no motion signal
    kicked = rep.ball_v[a:b] >= cfg.th_ball_move
    ball_out = rep.ball_out[a:b]  # tracked clearly beyond the touchline
    in_restart = ev.in_restart[a:b]
    e_team = ev.e_team[a:b]
    tag_score = ev.tag_score[a:b]

    # ---- dead / alive machine -------------------------------------------
    dead = np.zeros(m, dtype=np.bool_)
    age = 0
    enter_r = 0  # consecutive frames of restart-window dead evidence
    enter_m = 0  # consecutive frames of motion-lull dead evidence
    enter_b = 0  # consecutive frames of ball-outside-pitch dead evidence
    exit_p = 0  # consecutive frames of alive evidence
    restart_over_until = -1  # a restart window closed recently -> allow exit
    state_dead = False

    for i in range(m):
        enter_b = enter_b + 1 if ball_out[i] else 0
        if not state_dead:
            dead_restart = (
                in_restart[i] and not kicked[i] and motion[i] <= cfg.th_motion_high
            )
            dead_motion = (
                motion_known[i] and motion[i] < cfg.th_motion_low and not kicked[i]
            )
            enter_r = enter_r + 1 if dead_restart else 0
            enter_m = enter_m + 1 if dead_motion else 0
            if age >= cfg.t_min_state_frames and (
                enter_r >= cfg.t_enter_frames
                or enter_m >= cfg.t_enter_frames * 2
                or enter_b >= cfg.t_enter_frames
            ):
                state_dead = True
                dead[i:] = True
                age = 0
                enter_r = enter_m = 0
            else:
                age += 1
        else:
            if i > 0 and in_restart[i - 1] and not in_restart[i]:
                restart_over_until = i + cfg.t_exit_frames
            alive_evidence = enter_b < cfg.t_enter_frames and (
                kicked[i]
                or (motion_known[i] and motion[i] > cfg.th_motion_high)
                or i <= restart_over_until
            )
            exit_p = exit_p + 1 if alive_evidence else 0
            if age >= cfg.t_min_state_frames and exit_p >= cfg.t_exit_frames:
                state_dead = False
                dead[i:] = False
                age = 0
                exit_p = 0
            else:
                age += 1

    # ---- possession machine ---------------------------------------------
    state: int | None = None
    age = 0
    run = 0  # consecutive fresh-evidence frames pointing away from `state`
    since_restart = 10**9  # frames since the last dead -> alive edge

    for i in range(m):
        if dead[i]:
            out[a + i] = DEAD
            continue

        if i > 0 and dead[i - 1]:
            # just back from a dead ball: dwell already elapsed, answer fast
            age = cfg.t_min_poss_frames
            run = 0
            since_restart = 0

        e = e_team[i]
        if np.isfinite(e):
            s = float(np.clip(e + cfg.w_tag * tag_score[i], -1.0, 1.0))
            if state is None:
                if abs(s) > 0.15:
                    state = HOME if s > 0 else AWAY
                    age = 0
                    run = 0
            else:
                if (state == HOME and s < -cfg.th_switch) or (
                    state == AWAY and s > cfg.th_switch
                ):
                    run += 1
                else:
                    run = 0
                # right after a restart, cap the run needed to take over (the
                # held team usually conceded the stoppage); normal turnovers
                # keep the full t_switch_frames requirement
                need = (
                    min(cfg.t_switch_frames, cfg.t_restart_flip_frames)
                    if since_restart < cfg.t_min_poss_frames
                    else cfg.t_switch_frames
                )
                if age >= cfg.t_min_poss_frames and run >= need:
                    state = 1 - state
                    age = 0
                    run = 0
                    since_restart = 10**9  # takeover done
        # No fresh evidence: state and run hold (ball lost -> keep last team).

        out[a + i] = DEAD if state is None else state
        age += 1
        since_restart += 1
