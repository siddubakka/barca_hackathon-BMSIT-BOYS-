"""Load the competition match assets.

Reads (no ground truth anywhere in this module):

  * ``BAR-ATH_tactical_atd.xml``  -- FIFA/EPTS metadata: teams, players, the
    channel -> slot mapping of the raw file.
  * ``BAR-ATH_tactical_atd.txt``  -- raw tracking, one line per video frame:
    ``frame_id:x,y,speed;...;x,y,speed:x,y,speed`` with literal ``NaN`` for
    missing values.  Exactly two ':' separators; 46 player groups + ball.
  * ``BAR-ATH_tactical_homography.csv`` -- per-frame 3x3 matrix (normalised
    pitch -> image pixels), reprojection error, validity flag.
  * ``BAR-ATH_tactical_smart_tagging.xml`` -- SportsCode instances with code
    and label groups; times are video seconds, frame = round(sec * 25).

Alignment invariants (verified against the official starter/ loaders):

  * ATD ``frame_id`` is 1-based and contiguous -> submission row = frame_id-1
  * Smart Tagging intervals are half-open ``[round(start*25), round(end*25))``
  * homography row order == submission row order (``frame`` = 0..199318)

Parsed tracking is cached as an .npz next to the data (outside the repo) so
repeated runs do not re-parse 195 MB of text.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import FPS, GAME, PITCH_LENGTH_M, PITCH_WIDTH_M

GROUP_SEP = ";"
FIELD_SEP = ":"
MISSING = "N"  # a group starting with 'N' means NaN (baseline.py MISSING)


def _local(tag: str) -> str:
    """Strip an XML namespace if present."""
    return tag.rsplit("}", 1)[-1]


@dataclass
class Track:
    """Per-frame tracking arrays for one source."""

    n: int
    fps: float
    ball_x: np.ndarray  # (n,) normalised, NaN when missing
    ball_y: np.ndarray
    ball_s: np.ndarray  # (n,) m/s as exported (may be NaN/garbage; unused by baseline)
    px: np.ndarray  # (n, slots) normalised, NaN when missing
    py: np.ndarray
    ps: np.ndarray  # (n, slots) m/s speed channel
    team: np.ndarray  # (slots,) int8: 0 = home, 1 = away
    team_names: tuple[str, str]
    pitch: tuple[float, float] = (PITCH_LENGTH_M, PITCH_WIDTH_M)
    meta: dict = field(default_factory=dict)

    @property
    def seen_ball(self) -> np.ndarray:
        return np.isfinite(self.ball_x) & np.isfinite(self.ball_y)

    @property
    def n_slots(self) -> int:
        return self.px.shape[1]


@dataclass
class Homography:
    frame: np.ndarray  # (n,)
    matrices: np.ndarray  # (n, 3, 3) NaN where invalid
    reproj: np.ndarray  # (n,)
    valid: np.ndarray  # (n,) bool

    @property
    def n(self) -> int:
        return len(self.frame)


@dataclass
class Tag:
    code: str
    start_f: int
    end_f: int  # exclusive
    tags: dict[str, str]  # lowercased group -> text


# --------------------------------------------------------------------------
# metadata XML
# --------------------------------------------------------------------------

def load_metadata(path: str) -> dict:
    """Return per-slot team ids, team names, and slot/track counts.

    Slot order == raw-file group order.  We recover it from the
    DataFormatSpecification (PlayerChannelRef -> PlayerChannel -> playerId ->
    Player -> teamId) when present, falling back to <Player> document order,
    which is what starter/baseline.py assumes.
    """
    root = ET.parse(path).getroot()

    teams = [e for e in root.iter() if _local(e.tag) == "Team"]
    if len(teams) != 2:
        raise ValueError(f"{path}: expected 2 teams, found {len(teams)}")
    team_ids = [t.get("id") for t in teams]
    names = tuple((t.findtext("Name") or "").strip() for t in teams)  # (home, away)

    players = [e for e in root.iter() if _local(e.tag) == "Player"]
    pid_to_team = {}
    for p in players:
        tid = p.get("teamId")
        pid_to_team[p.get("id")] = 0 if tid == team_ids[0] else 1

    # playerChannelId -> playerId
    pchannels = {
        e.get("id"): e.get("playerId")
        for e in root.iter()
        if _local(e.tag) == "PlayerChannel"
    }

    # slot order from the data-format spec
    slots: list[int | None] = []
    ball_slot = None
    for spec in (e for e in root.iter() if _local(e.tag) == "DataFormatSpecification"):
        for split in spec.iter():
            if _local(split.tag) != "SplitRegister":
                continue
            refs = [c for c in split if _local(c.tag) == "PlayerChannelRef"]
            ball_refs = [c for c in split if _local(c.tag) == "BallChannelRef"]
            if refs:
                pid = pchannels.get(refs[0].get("playerChannelId"))
                slots.append(pid_to_team.get(pid))
            elif ball_refs:
                ball_slot = len(slots)

    if not slots:
        slots = [pid_to_team.get(p.get("id")) for p in players]  # document order

    if any(s is None for s in slots):
        raise ValueError(f"{path}: could not map every track slot to a team")

    frame_rate = None
    for e in root.iter():
        if _local(e.tag) == "FrameRate" and (e.text or "").strip():
            frame_rate = float(e.text.strip())
            break

    return {
        "team": np.asarray(slots, dtype=np.int8),
        "team_names": names,
        "ball_slot": ball_slot,
        "fps": frame_rate or FPS,
        "n_players": len(players),
    }


# --------------------------------------------------------------------------
# raw ATD txt
# --------------------------------------------------------------------------

def _cache_path(raw_path: str) -> str:
    d = os.path.dirname(os.path.abspath(raw_path))
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(d, "..", "cache", f"{stem}.npz")


def load_atd(meta_path: str, raw_path: str, use_cache: bool = True) -> Track:
    """Parse the raw ATD text file into arrays (frame_id - 1 indexing)."""
    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        try:
            mtime = os.path.getmtime(raw_path)
            if os.path.getmtime(cache) >= mtime:
                return _track_from_npz(cache, meta_path)
        except OSError:
            pass

    meta = load_metadata(meta_path)
    team = meta["team"]
    n_slots = len(team)

    # pass 1: line count (records are contiguous from 1 on this delivery)
    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)

    ball_x = np.full(n, np.nan, dtype=np.float32)
    ball_y = np.full(n, np.nan, dtype=np.float32)
    ball_s = np.full(n, np.nan, dtype=np.float32)
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n, n_slots), np.nan, dtype=np.float32)

    n_bad_lines = 0
    n_slot_mismatch = 0
    n_frames_seen = 0

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                n_bad_lines += 1
                continue
            try:
                row = int(parts[0]) - 1  # frame_id -> 0-based submission row
            except ValueError:
                n_bad_lines += 1
                continue
            if row < 0 or row >= n:
                n_bad_lines += 1
                continue
            n_frames_seen += 1

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                n_slot_mismatch += 1
            for i, g in enumerate(groups):
                if i >= n_slots:
                    break
                if g[0] == MISSING:
                    continue
                c1 = g.find(",")
                if c1 < 0:
                    continue
                c2 = g.find(",", c1 + 1)
                if c2 < 0:
                    continue
                try:
                    px[row, i] = float(g[:c1])
                    py[row, i] = float(g[c1 + 1 : c2])
                    ps[row, i] = float(g[c2 + 1 :])
                except ValueError:
                    pass

            g = parts[2]
            if g and g[0] != MISSING:
                c1 = g.find(",")
                if c1 >= 0:
                    c2 = g.find(",", c1 + 1)
                    if c2 >= 0:
                        try:
                            ball_x[row] = float(g[:c1])
                            ball_y[row] = float(g[c1 + 1 : c2])
                            ball_s[row] = float(g[c2 + 1 :])
                        except ValueError:
                            pass

    tr = Track(
        n=n,
        fps=meta["fps"],
        ball_x=ball_x,
        ball_y=ball_y,
        ball_s=ball_s,
        px=px,
        py=py,
        ps=ps,
        team=team,
        team_names=meta["team_names"],
        meta={
            "bad_lines": n_bad_lines,
            "slot_mismatch": n_slot_mismatch,
            "frames_seen": n_frames_seen,
            "raw": os.path.abspath(raw_path),
        },
    )
    _save_npz(cache, tr, meta_path)
    return tr


def _save_npz(cache: str, tr: Track, meta_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(
            cache,
            ball_x=tr.ball_x,
            ball_y=tr.ball_y,
            ball_s=tr.ball_s,
            px=tr.px,
            py=tr.py,
            ps=tr.ps,
            team=tr.team,
            team_names=np.asarray(tr.team_names),
            fps=np.float64(tr.fps),
            meta_mtime=np.float64(os.path.getmtime(meta_path)),
        )
    except OSError:
        pass  # caching is best-effort


def _track_from_npz(cache: str, meta_path: str) -> Track:
    z = np.load(cache, allow_pickle=False)
    if float(z["meta_mtime"]) != os.path.getmtime(meta_path):
        raise ValueError("cache stale")  # caller falls back to parsing
    names = tuple(str(x) for x in z["team_names"])
    return Track(
        n=len(z["ball_x"]),
        fps=float(z["fps"]),
        ball_x=z["ball_x"],
        ball_y=z["ball_y"],
        ball_s=z["ball_s"],
        px=z["px"],
        py=z["py"],
        ps=z["ps"],
        team=z["team"],
        team_names=names,
        meta={"cache": cache},
    )


# --------------------------------------------------------------------------
# homography
# --------------------------------------------------------------------------

MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def load_homography(path: str) -> Homography:
    table = pd.read_csv(path)
    matrices = (
        table[MATRIX_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    )
    valid = table["valid"].to_numpy() == 1
    return Homography(
        frame=table["frame"].to_numpy(),
        matrices=matrices,
        reproj=table["reprojection_error"].to_numpy(dtype=np.float64),
        valid=valid,
    )


# --------------------------------------------------------------------------
# smart tagging (SportsCode)
# --------------------------------------------------------------------------

def load_smart_tagging(path: str, fps: float = FPS) -> list[Tag]:
    """Parse Smart Tagging instances into frame intervals.

    Time base: seconds of the tactical video -> ``round(sec * fps)``.
    Intervals are half-open [start_f, end_f).  When a group appears twice in
    one instance the last occurrence wins (same as starter/loader).
    """
    root = ET.parse(path).getroot()
    tags: list[Tag] = []
    for inst in root.iter():
        if _local(inst.tag) != "instance":
            continue
        start_s = inst.findtext("start")
        end_s = inst.findtext("end")
        code = (inst.findtext("code") or "").strip()
        if start_s is None or end_s is None:
            continue
        labels: dict[str, str] = {}
        for lab in inst.iter():
            if _local(lab.tag) != "label":
                continue
            g = (lab.findtext("group") or "").strip().lower()
            t = (lab.findtext("text") or "").strip()
            if g:
                labels[g] = t
        start_f = int(round(float(start_s) * fps))
        end_f = int(round(float(end_s) * fps))
        tags.append(Tag(code=code, start_f=start_f, end_f=max(end_f, start_f), tags=labels))
    return tags


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def asset_paths(data_dir: str, game: str = GAME, source: str = "tactical") -> dict[str, str]:
    """Resolve asset paths under ``data_dir`` (supports a tactical/ subdir)."""
    base = data_dir
    if os.path.isdir(os.path.join(data_dir, "tactical")):
        base = os.path.join(data_dir, "tactical")
    stem = f"{game}_{source}"  # e.g. BAR-ATH_tactical (games.py SOURCES stem)
    paths = {
        "atd_meta": os.path.join(base, f"{stem}_atd.xml"),
        "atd_raw": os.path.join(base, f"{stem}_atd.txt"),
        "homography": os.path.join(base, f"{stem}_homography.csv"),
        "smart_tagging": os.path.join(base, f"{stem}_smart_tagging.xml"),
        "video": os.path.join(base, f"{stem}.mp4"),
    }
    missing = [k for k, p in paths.items() if k != "video" and not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "missing competition assets under %s: %s" % (base, ", ".join(missing))
        )
    return paths
