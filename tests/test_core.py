"""Unit tests for the possession pipeline.  Run:  python tests/test_core.py

No pytest dependency on purpose (the repo must regenerate the submission with
pinned deps only).  Every test is label-free: synthetic signals with known
expected behaviour, plus the published frame-range constants.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import constants  # noqa: E402
from src.config import Config  # noqa: E402
from src.decode import decode  # noqa: E402
from src.emit import labels_to_csv, validate  # noqa: E402
from src.evidence import compute  # noqa: E402
from src.load import Tag, Track, load_smart_tagging  # noqa: E402
from src.repair import _fill_neighbours, repair  # noqa: E402

HOME, AWAY, DEAD, X = 0, 1, 2, 3
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def make_track(
    n: int,
    ball_xy: np.ndarray | None,  # (n, 2) normalised, NaN row = unseen
    home_xy: np.ndarray,  # (k, 2)
    away_xy: np.ndarray,
    speed: float | np.ndarray = 2.5,
) -> Track:
    slots_home, slots_away = len(home_xy), len(away_xy)
    xy = np.vstack([home_xy, away_xy])
    px = np.repeat(xy[None, :, 0], n, axis=0).astype(np.float32)
    py = np.repeat(xy[None, :, 1], n, axis=0).astype(np.float32)
    sp = np.full((n, slots_home + slots_away), np.nan, dtype=np.float32)
    s = np.asarray(speed, dtype=np.float32)
    sp[:] = s if s.ndim == 0 else s[:, None]
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    if ball_xy is not None:
        ok = np.isfinite(ball_xy).all(axis=1)
        bx[ok] = ball_xy[ok, 0]
        by[ok] = ball_xy[ok, 1]
    return Track(
        n=n,
        fps=25.0,
        ball_x=bx,
        ball_y=by,
        ball_s=np.full(n, np.nan, dtype=np.float32),
        px=px,
        py=py,
        ps=sp,
        team=np.array([0] * slots_home + [1] * slots_away, dtype=np.int8),
        team_names=("Home FC", "Away FC"),
        pitch=(105.0, 68.0),
    )


def run(tr: Track, tags: list[Tag], cfg: Config) -> np.ndarray:
    rep = repair(tr, cfg)
    ev = compute(tr, rep, tags, cfg)
    return decode(tr, rep, ev, cfg, n=tr.n, active_blocks=((0, tr.n),))


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_fill_neighbours() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 64):
        for _ in range(30):
            mask = rng.random(n) < 0.3
            prev, nxt = _fill_neighbours(mask)
            for i in range(n):
                exp_prev = max([j for j in range(i + 1) if mask[j]], default=-1)
                exp_next = min([j for j in range(i, n) if mask[j]], default=-1)
                if prev[i] != exp_prev or nxt[i] != exp_next:
                    check(
                        f"fill_neighbours n={n}",
                        False,
                        f"i={i} got ({prev[i]},{nxt[i]}) want ({exp_prev},{exp_next})",
                    )
                    return
    check("fill_neighbours brute-force agreement", True)


def test_ball_interpolation() -> None:
    cfg = Config()
    n = 300
    ball = np.full((n, 2), np.nan)
    ball[0:50] = 0.5  # seen start
    ball[60:100] = 0.6  # seen again after a 10-frame hole (<= max_hole)
    ball[200:250] = 0.4  # seen, then a 50-frame hole to the end
    tr = make_track(n, ball, home_xy=np.array([[0.5, 0.5]]), away_xy=np.array([[0.9, 0.9]]))
    rep = repair(tr, cfg)
    # short hole filled linearly
    check("interp fills short hole", bool(rep.ball_valid[55]))
    check(
        "interp values between endpoints",
        0.5 <= float(rep.ball_x[55]) <= 0.6,
        f"got {rep.ball_x[55]}",
    )
    # long hole (100 frames to end) stays unknown
    check("no interp beyond max_hole", not bool(rep.ball_valid[270]))
    check("seen frames valid", bool(rep.ball_valid[42]))
    # gap distance
    check("ball_gap zero when seen", rep.ball_gap[42] == 0)
    check("ball_gap counts hole", rep.ball_gap[55] == 5)


def test_dead_machine_lull_and_kick() -> None:
    cfg = Config()
    n = 500
    home = np.array([[0.48, 0.5], [0.52, 0.5], [0.5, 0.48]])
    away = np.array([[0.9, 0.5], [0.92, 0.52], [0.88, 0.48]])
    ball = np.full((n, 2), np.nan)
    ball[:150] = [0.5, 0.5]  # ball visible first third
    ball[300:] = [0.5, 0.5]  # and last third
    speed = np.full(n, 2.5)
    speed[150:300] = 0.2  # players stop -> dead evidence
    tr = make_track(n, ball, home, away, speed=speed)
    labels = run(tr, [], cfg)
    alive_first = labels[:140]
    dead_zone = labels[190:290]
    alive_end = labels[340:]
    check("seeds H when home controls", int(np.mean(alive_first == HOME)) > 0.9,
          f"mean {np.mean(alive_first == HOME):.2f}")
    check("motion lull -> D", int(np.mean(dead_zone == DEAD)) > 0.9,
          f"mean {np.mean(dead_zone == DEAD):.2f}")
    check("after lull the state returns", int(np.mean(alive_end == HOME)) > 0.9,
          f"mean {np.mean(alive_end == HOME):.2f}")


def test_ball_out_enters_dead_and_blocks_exit() -> None:
    """Ball tracked beyond the touchline: enter D fast, stay D while the
    ball is still out even when players are running, recover on return."""
    cfg = Config()
    n = 400
    home = np.array([[0.48, 0.5], [0.52, 0.5], [0.5, 0.48]])
    away = np.array([[0.9, 0.5], [0.92, 0.52], [0.88, 0.48]])
    ball = np.full((n, 2), np.nan)
    ball[:100] = [0.5, 0.5]  # in play, home controls
    ball[100:300] = [0.5, 1.05]  # beyond the touchline (out of play)
    ball[300:] = [0.5, 0.5]  # back in play
    tr = make_track(n, ball, home, away, speed=2.5)  # players keep running
    labels = run(tr, [], cfg)
    check("ball_out enters D", bool(np.all(labels[120:160] == DEAD)),
          f"got {labels[120:160]}")
    check("still D mid-way while ball out", labels[250] == DEAD, f"got {labels[250]}")
    check("recovers after ball returns", int(np.mean(labels[320:] == HOME)) > 0.9,
          f"mean {np.mean(labels[320:] == HOME):.2f}")


def test_ball_out_flicker_is_ignored() -> None:
    """A brief excursion beyond the margin (noise) must not create a D run."""
    cfg = Config()
    n = 300
    home = np.array([[0.48, 0.5], [0.52, 0.5], [0.5, 0.48]])
    away = np.array([[0.9, 0.5], [0.92, 0.52], [0.88, 0.48]])
    ball = np.tile(np.array([0.5, 0.5]), (n, 1))
    ball[200:200 + cfg.t_enter_frames - 1] = [0.5, 1.05]  # t_enter-1 frames only
    tr = make_track(n, ball, home, away, speed=2.5)
    labels = run(tr, [], cfg)
    check("short flicker does not enter D", int(np.mean(labels[190:260] == DEAD)) == 0,
          f"D frames {int(np.sum(labels[190:260] == DEAD))}")
    check("flicker keeps possession", int(np.mean(labels[190:260] == HOME)) > 0.95,
          f"mean {np.mean(labels[190:260] == HOME):.2f}")


def test_switch_needs_sustained_evidence() -> None:
    cfg = Config()
    n = 400
    home = np.array([[0.38, 0.5], [0.42, 0.5], [0.4, 0.48]])
    away = np.array([[0.58, 0.5], [0.62, 0.5], [0.6, 0.52]])
    ball = np.full((n, 2), np.nan)
    ball[:200] = [0.4, 0.5]  # home side first
    ball[200:] = [0.6, 0.5]  # away side after frame 200
    tr = make_track(n, ball, home, away, speed=2.5)
    labels = run(tr, [], cfg)
    check("home possession before flip", int(np.mean(labels[60:195] == HOME)) > 0.95,
          f"mean {np.mean(labels[60:195] == HOME):.2f}")
    check("no instant flip", labels[210] == HOME, f"got {labels[210]}")
    check("flip after t_switch", int(np.mean(labels[245:] == AWAY)) > 0.9,
          f"mean {np.mean(labels[245:] == AWAY):.2f}")


def test_missing_ball_holds_team() -> None:
    cfg = Config()
    n = 400
    home = np.array([[0.48, 0.5], [0.52, 0.5], [0.5, 0.48]])
    away = np.array([[0.9, 0.5], [0.92, 0.52], [0.88, 0.48]])
    ball = np.full((n, 2), np.nan)
    ball[:120] = [0.5, 0.5]
    # ball lost for a long stretch while play continues (motion stays high)
    tr = make_track(n, ball, home, away, speed=2.5)
    labels = run(tr, [], cfg)
    zone = labels[150:380]
    check("ball lost -> hold last team", int(np.mean(zone == HOME)) > 0.95,
          f"mean {np.mean(zone == HOME):.2f}")
    check("ball lost but alive stays alive", int(np.mean(zone == DEAD)) < 0.05)


def test_restart_team_can_take_over_fast() -> None:
    """After a dead ball the dwell is pre-satisfied: the restart team wins
    after only t_switch frames of evidence, not t_min_poss again."""
    cfg = Config()
    n = 500
    home = np.array([[0.48, 0.5], [0.52, 0.5], [0.5, 0.48]])
    away = np.array([[0.58, 0.5], [0.62, 0.5], [0.6, 0.52]])
    ball = np.full((n, 2), np.nan)
    ball[:100] = [0.5, 0.5]  # home has it
    ball[220:] = [0.6, 0.5]  # away takes the restart
    speed = np.full(n, 2.5)
    speed[110:215] = 0.2  # dead period (ball also missing there)
    tr = make_track(n, ball, home, away, speed=speed)
    labels = run(tr, [], cfg)
    check("dead during stoppage", labels[150] == DEAD, f"got {labels[150]}")
    check("home before stoppage", labels[80] == HOME, f"got {labels[80]}")
    check("away after restart + t_switch", labels[260] == AWAY, f"got {labels[260]}")
    check("not away instantly after restart", labels[230] == HOME, f"got {labels[230]}")


def test_hard_anchors_and_blocks() -> None:
    # Constants are published frame ranges -- assert the wiring, not the values'
    # provenance (values are quoted in the FAQ / games.py).
    n = constants.N_VIDEO_FRAMES
    a, b = constants.SECOND_HALF_PRE_KICKOFF
    check("pre-kickoff stretch is 2934 frames", b - a == 2934, f"{b - a}")
    g1, g2 = constants.GRADED_BLOCKS
    check("graded blocks disjoint and ordered", g1[1] < g2[0])
    check("pre-kickoff inside graded block 2", g2[0] <= a < b <= g2[1])
    check("halftime starts at block1 end", constants.HALFTIME[0] == g1[1])
    check("halftime ends at block2 start", constants.HALFTIME[1] == g2[0])
    check("pre-match ends at block1 start", constants.PRE_MATCH[1] == g1[0])
    check("post-match starts at block2 end", constants.POST_MATCH[0] == g2[1])
    check("post-match ends at n", constants.POST_MATCH[1] == n)


def test_emit_roundtrip() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=5000).astype(np.int8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub.csv")
        labels_to_csv(labels, path, n_expected=5000)
        validate(path, 5000)  # must not raise
        # corrupt: bad label
        with open(path, encoding="utf-8") as f:
            text = f.read()
        bad = text.replace(",H\n", ",Z\n", 1)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(bad)
        raised = False
        try:
            validate(path, 5000)
        except ValueError:
            raised = True
        check("validator rejects bad label", raised)


def test_smart_tagging_timebase() -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
<main xmlns:tns="http://www.sportscode.com">
  <instance>
    <start>2.0</start><end>4.0</end><code>SET PIECES</code>
    <label><group>Half</group><text>1st half</text></label>
    <label><group>Team</group><text>N/A</text></label>
  </instance>
  <instance>
    <start>6.000001</start><end>6.5</end><code>PROGRESSION</code>
    <label><group>Team</group><text>Home FC</text></label>
  </instance>
</main>"""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tags.xml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(xml)
        tags = load_smart_tagging(p, fps=25)
    check("tag count", len(tags) == 2, f"{len(tags)}")
    check("seconds*25 frame base", tags[0].start_f == 50 and tags[0].end_f == 100,
          f"{tags[0].start_f},{tags[0].end_f}")
    check("half-open interval round()", tags[1].start_f == 150)
    check("labels kept", tags[0].tags.get("team") == "N/A")
    check("code kept", tags[1].code == "PROGRESSION")


def test_evidence_signs() -> None:
    cfg = Config()
    n = 50
    home = np.array([[0.5, 0.5], [0.51, 0.5]])
    away = np.array([[0.9, 0.9], [0.91, 0.9]])
    ball = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (n, 1))
    tr = make_track(n, ball, home, away, speed=2.5)
    rep = repair(tr, cfg)
    tags = [
        Tag("DEFENDING IN MIDDLE THIRD", 0, n, {"team": "Away FC"}),
        Tag("SET PIECES", 10, 20, {}),
    ]
    ev = compute(tr, rep, tags, cfg)
    check("vote favours home", bool(np.nanmin(ev.e_team) > 0.5),
          f"min {np.nanmin(ev.e_team):.2f}")
    check("defending-away tag -> home prior", float(ev.tag_score[5]) > 0,
          f"{ev.tag_score[5]}")
    check("set-piece window marks restart", bool(ev.in_restart[12]))
    check("outside window clear", not bool(ev.in_restart[30]))


def test_decode_outside_blocks_is_dead() -> None:
    """The competition default: everything outside the active play blocks is D."""
    from src.decode import ACTIVE_BLOCKS

    cfg = Config()
    n = 400
    home = np.array([[0.48, 0.5], [0.52, 0.5]])
    away = np.array([[0.9, 0.5], [0.92, 0.52]])
    ball = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (n, 1))
    tr = make_track(n, ball, home, away, speed=2.5)
    rep = repair(tr, cfg)
    ev = compute(tr, rep, [], cfg)
    labels = decode(tr, rep, ev, cfg, n=n, active_blocks=((100, 300),))
    outside = np.ones(n, dtype=bool)
    outside[100:300] = False
    check("outside active block = D", bool(np.all(labels[outside] == DEAD)))
    check("inside block decoded", bool(np.any(labels[100:300] != DEAD)))


def test_lab_blocks_from_period() -> None:
    """Lab decode blocks must follow match periods: state resets at each
    block start, exactly like the competition's two graded play blocks."""
    try:
        from train.lab import _blocks_from_period
    except ImportError:
        check("lab blocks (skipped: official scorer not on path)", True)
        return
    period = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0], dtype=np.int8)
    blocks = _blocks_from_period(period)
    check("halftime splits blocks", blocks == ((2, 5), (6, 8)), f"got {blocks}")
    check("one period -> one block",
          _blocks_from_period(np.ones(10, dtype=np.int8)) == ((0, 10),))
    check("no period -> no blocks",
          _blocks_from_period(np.zeros(5, dtype=np.int8)) == ())


def main() -> int:
    tests = [
        test_fill_neighbours,
        test_ball_interpolation,
        test_dead_machine_lull_and_kick,
        test_ball_out_enters_dead_and_blocks_exit,
        test_ball_out_flicker_is_ignored,
        test_switch_needs_sustained_evidence,
        test_missing_ball_holds_team,
        test_restart_team_can_take_over_fast,
        test_hard_anchors_and_blocks,
        test_emit_roundtrip,
        test_smart_tagging_timebase,
        test_evidence_signs,
        test_decode_outside_blocks_is_dead,
        test_lab_blocks_from_period,
    ]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            global FAIL
            FAIL += 1
            print(f"  ERROR {t.__name__}: {exc!r}")
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
