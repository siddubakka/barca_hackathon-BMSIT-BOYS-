# Technical Report — mtah2027-possession

Competition: **More than a Hack 2027 — Phase 1: Ball Possession**
(Metrica Sports / FC Barcelona). Source: `BAR-ATH` tactical tracking.
Ranking metric: `spell_f1_mean`, tie-break `macro_f1`.

---

## 1. Problem and ground-truth semantics

The task is to label every one of the 199,319 video frames (25 fps) as

* `H` / `A` — the home (Barcelona) / away (Atletic Club) team is in possession,
* `D` — the ball is **not in play** (dead),
* `X` — abstain (unused by this submission; every graded frame gets a real
  answer, so `n_abstain_on_graded = 0`).

Per the brief, ground truth is derived mechanically from corrected tracking:
dead ⇒ `D`, otherwise the team tracking says has the ball; a *possession* is a
maximal run of one team **with the ball alive**, so every out-of-play spell
splits possessions even when the same team restarts. Ranked on mean temporal
IoU F1 over possessions (0.3…0.7), which makes **fragmentation** the dominant
failure mode: too many predicted spells precision-crush the score, while missed
dead-ball splits merge GT spells and cost recall.

Two facts from the data dominate the design:

1. The ball is tracked on only **56.9 % of frames** (59.3 % graded). A pipeline
   that equates "no ball" with "dead" (as the starter baseline does, via its
   ±50-frame vote window) over-fragments possessions.
2. Frames **110808–113741 are graded and all `D`** (second half is graded from
   110808 but kicks off at 113742) — published in the FAQ, hard-anchored.

## 2. Assets: used vs dropped

| Asset | Status | Why |
|---|---|---|
| `BAR-ATH_tactical_atd.txt` (195 MB raw tracking) | **used** | ball + 46 player/tracklet slots per frame (`x,y,speed`), the primary signal |
| `BAR-ATH_tactical_atd.xml` (metadata) | **used** | team/slot mapping (24 home, 22 away slots), frame rate, pitch size |
| `BAR-ATH_tactical_smart_tagging.xml` (638 instances) | **used** | `SET PIECES` (59) → restart windows for dead-ball logic; phase codes with `Team` labels → weak possession prior |
| Published frame ranges (FAQ / `games.py`) | **used** | graded blocks, pre-second-half dead stretch, hard anchors |
| Official `evaluation/` scorer | **used** | validation + the single public-window pipeline check |
| `BAR-ATH_tactical_homography.csv` (23 MB) | parsed, **dropped** | 100 % valid, median reprojection 0.24 px — but every decision is made in pitch coordinates; the homography only feeds image-space work (video), which we dropped |
| `BAR-ATH_tactical.mp4` (4.9 GB broadcast video) | **dropped** | no CV/video model in scope; broadcast↔tactical alignment is risky and unverifiable without labeling; watching video is permitted for debugging but nothing here needs it |
| Starter `baseline.py` | reference only | reproduced to verify our ATD parser (ball located on 56.9 % of frames, `H = Barcelona, A = Atletic Club` — matches the brief exactly) |

No competition match data is stored in this repository (competition rule).

## 3. Method

```
load.py → repair.py → evidence.py → decode.py → emit.py
```

### 3.1 Load & alignment (`src/load.py`)

* ATD `frame_id` is 1-based ⇒ submission row = `frame_id − 1` (verified against
  the official loader; raw records are contiguous, 199,319 lines).
* Smart Tagging times are video seconds ⇒ frame = `round(sec × 25)`, intervals
  half-open `[start, end)` (the official `check_time_base` semantics: both
  kickoffs fall inside `SET PIECES` windows — verified).
* Parsed once and cached as `.npz` next to the data (outside the repo).
* Diagnostics printed (aggregates only): ball seen 56.9 % overall / 59.3 %
  graded, player speed channel median 1.34 m/s (max 11.3), homography 100 %
  valid.

### 3.2 Repair (`src/repair.py`)

* **Ball interpolation**: linear fill only across holes ≤ `max_hole_frames`
  (2 s default); longer holes stay unknown — never invent a ball across a
  multi-second blackout.
* **Ball speed**: geometry-derived (metres between near-consecutive sightings,
  ≤ 6 frames apart) with a ±3-frame rolling max so a kick stays visible for a
  few frames. The shipped `ball_s` channel is capped at ~7 m/s and was
  therefore not trusted for kick detection.
* **Ball out of play**: a raw sighting beyond the touchline (1.5 % margin in
  normalised pitch coordinates) marks the ball as out of play. Measured on the
  competition match: 55 runs, median 23 frames, 97 % of the frames sit in runs
  ≥ 8 frames — sustained events (throw-ins, goal kicks), not tracking jitter.
  Half of them fall inside Smart Tagging restart windows; the other half are
  genuine out-of-play moments the tags do not cover.
* **Motion**: median speed of tracked players per frame; frames with < 3
  tracked players carry *no* motion signal (they must not masquerade as a
  lull).

### 3.3 Evidence (`src/evidence.py`)

* **Tracking vote**: every tracked player votes for its team with Gaussian
  weight `exp(−d²/2σ²)` (σ = 2.5 m tuned, capped at `r_max` = 30 m) against the
  repaired ball position ⇒ `e_team ∈ [−1,1]`. This is a smoothed, margin-aware
  version of the baseline's hard nearest-player argmin: a defender stepping
  close to a ball still controlled by the opponent does not instantly flip the
  evidence, but a clear interception does.
* **Smart Tagging priors**:
  * `SET PIECES` windows ⇒ `in_restart` (ball presumed dead until the kick);
  * phase codes with unambiguous possession semantics contribute a weak team
    prior with weight `w_tag = 0.30`:
    `OWN = {PROGRESSION, BUILD UP, LONG BALL, CREATING CHANCES, ATTACKING
    TRANSITION, SET PIECES}` (team named has the ball), `DEF = {DEFENSIVE
    TRANSITION, DEFENDING IN *}` (team named is defending ⇒ inverted).
  * Codes whose direction is ambiguous from documentation alone
    (`BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, `GOALS`)
    are **not** used. The prior only *biases* the tracking vote when fresh vote
    exists — it never decides alone (tags are ~20 s windows and can be stale at
    a transition).

### 3.4 Decode (`src/decode.py`) — two state machines

**Dead/alive** (hysteresis + minimum dwell both ways):

* enter `D` on a restart window with the ball not yet kicked (fast: `t_enter`
  frames), on the ball being tracked outside the touchline for `t_enter`
  frames, or on a sustained motion lull (`2 × t_enter` frames);
* leave `D` on a kick (`ball_v ≥ th_ball_move`), players running again
  (`motion > th_motion_high`), or the restart window closing — but **not**
  while the ball is still outside the touchline (the ball cannot be in play
  beyond the line, so exit is suppressed until it comes back);
* `t_min_state_frames` prevents flicker in either direction.

**Possession** (updates only while alive):

* the team state **holds** whenever fresh evidence is missing — this is the
  "possession stays with the last team during loose/contested phases" rule and
  the main structural improvement over the baseline;
* a switch requires `t_switch_frames` of *sustained* fresh evidence against
  the current team (a controlled turnover) and never before
  `t_min_poss_frames` of dwell;
* a dead→alive transition counts as satisfied dwell, so the team taking the
  restart can take over after only `t_switch_frames` of evidence;
* before the first evidence of a block the label is `D` (no guess).

**Anchors**: frames outside the two active play blocks
(`[11185,90652)`, `[113742,190198)`) are emitted `D` — pre-match, halftime, the
published pre-kickoff stretch (110808–113741), post-match. All of that except
the published stretch is ungraded anyway; anchoring keeps the decoder from
reasoning about warm-up footage.

### 3.5 Emit (`src/emit.py`)

Writes `frame,label` CSV (199,319 rows, integer frames in order, UTF-8) and
re-validates the exact contract of `evaluation/validate_submission.py` before
the run reports success.

## 4. Training data sources & tuning protocol

**Rule followed**: no competition-match labels (public 5-min GT used only once,
at the end, as a pipeline check), no hand-labeling of the competition match.

**Practice lab — SkillCorner open data** (`train/lab.py`): 20 full matches,
≈ **1.24 M annotated frames (34.6 h** of play at the source 10 fps),
labelled by SkillCorner — the dataset provider — through their own
`phases_of_play` annotations; no label in this project was hand-written by
us. 6 used for tuning
(`1874553, 1886347, 1899585, 1925299, 1927964, 1953632`), 14 held out.

Lab ground truth (from SkillCorner's own annotations only):

| Lab label | Source |
|---|---|
| `X` | outside a match period (pre-match / halftime hole) |
| `H/A` | inside a `phases_of_play` possession phase (`team_in_possession_id`) |
| `D` | between phases, unless SkillCorner's per-frame `possession` still names a team (loose ball) — then that team wins |

Why this matches the competition's semantics: in-phase frames have the ball
detected 94 % of the time and the ball out of the pitch 0.8 % of the time;
gap frames have no possessor 95 % of the time and the ball out of the pitch
44 % of the time. Phases = "alive, team controls"; gaps = "not in play /
not controlled".

To make the lab exercise *the same code* under *similar noise*:

* resampled 10 fps → 25 fps (frame constants carry over unchanged);
* decoded in **period-derived blocks** (one per half, halftime between them),
  so the state reset at a block start and the D-default outside blocks —
  behaviour the competition match exercises at both kickoffs — are tuned
  under the same conditions they will run in;
* in-play ball visibility degraded from SkillCorner's ~90 % to **65 %**
  (the competition's estimated in-play visibility given 59.3 % graded
  visibility and ~20–25 % dead time), dropping contiguous bursts of 2–100
  frames;
* player speeds derived from positions at 10 fps (SkillCorner ships no speed
  channel), median-smoothed over 0.5 s, then resampled — the competition's
  ATD ships a real speed channel, so this is the closest legal stand-in;
* synthetic `SET PIECES` restart tags bracket each lab dead run's restart
  (−3.5 s / +1.5 s, ±0.5 s edge jitter, no team label — like the majority of
  real set-piece tags, which are `N/A`) so the restart code path is exercised
  with realistic imprecision.

**Selection** (`train/sweep.py`): coordinate descent over 12 hyperparameters
(2 passes, 78 evaluated configs) scoring **mean `spell_f1_mean`** over the
6 tune matches with mean `macro_f1` as tie-break — the competition's own
ranking rule, applied to legal data.

**Generalization check**: the frozen configuration is re-scored on the 14
held-out matches (report in §7).

## 5. Possession rules as implemented (brief → code)

| Brief / FAQ rule | Where |
|---|---|
| `D` iff ball not in play | restart windows, ball-outside-pitch and motion lull enter `D`; kick / running players / window close exit it, suppressed while the ball is still out (`decode._decode_block`) |
| ball missing ≠ dead; loose ball stays with last team | possession state holds while evidence is missing (`decode` possession machine) |
| change of possession needs controlled possession — a challenge, duel or momentum flip alone never switches team | `t_switch_frames` sustained opposite evidence + `t_min_poss_frames` dwell |
| restarts are dead until the kick, then the restart team can take over | `in_restart` window, dwell pre-satisfied at dead→alive |
| published 110808–113741 all `D` | `constants.SECOND_HALF_PRE_KICKOFF` hard anchor |
| ungraded frames unscored | emitted `D` (ball not in play there anyway); scorer ignores GT-`X` rows |

## 6. Hyperparameters (frozen from the lab)

Selected by `train/sweep.py` (coordinate descent, mean `spell_f1_mean` over
the 6 tune matches, macro tie-break) and frozen in `src/config.py`, which is
what `make_submission.py` runs. Full trial log: `train/sweep_results.json`.

| parameter | role | grid searched | chosen |
|---|---|---|---|
| `th_motion_low` | median player speed below → dead evidence | 0.4 / 0.6 / 0.8 / 1.0 / 1.3 | **0.4** |
| `th_motion_high` | median player speed above → alive (hysteresis) | 1.2 / 1.5 / 1.8 / 2.2 | **1.8** |
| `th_ball_move` | geometry ball speed meaning "kicked / in play" | 4 / 7 / 10 | **10** |
| `t_enter_frames` | consecutive frames of dead evidence to enter `D` (also ball-out) | 3 / 5 / 10 | **5** (default kept) |
| `t_exit_frames` | consecutive frames of alive evidence to leave `D` | 3 / 4 / 8 / 12 | **12** |
| `t_min_state_frames` | minimum dwell in either dead/alive state | 8 / 15 / 25 / 40 | **40** |
| `th_switch` | \|score\| needed to accumulate a team switch | 0.12 / 0.18 / 0.25 / 0.35 / 0.5 | **0.12** |
| `t_switch_frames` | sustained opposite evidence to switch teams | 12 / 25 / 40 / 60 (default 30 kept if best) | **30** (default kept) |
| `t_min_poss_frames` | minimum dwell before any switch | 25 / 50 / 75 / 125 / 200 | **75** (default kept) |
| `sigma_m` | Gaussian proximity kernel width | 2.5 / 3.5 / 5.0 / 7.0 | **2.5** |
| `r_max_m` | players beyond this distance do not vote | 20 / 30 / 45 | **30** (default kept) |
| `max_hole_frames` | ball interpolation trust radius | 12 / 25 / 50 / 100 / 250 | **50** (default kept) |

Not tunable in the lab (kept at first-principles defaults): `w_tag = 0.30`
(the lab has no phase tags to train it on; it only biases fresh tracking
votes and never decides alone).

## 7. Validation

Everything below ran with the frozen `src/config.py`; the public window was
scored exactly once, after freezing, and no parameter changed afterwards.

**Unit tests** — `python tests/test_core.py`: 47 checks, 0 failures
(neighbour/interpolation correctness, dead-machine lull→kick transitions,
ball-out entry / exit blocking / flicker rejection, sustained-evidence
switching, hold-through-lost-ball, restart fast takeover, published
frame-range wiring, lab block derivation, CSV contract, Smart Tagging time
base, evidence signs); `python -m compileall` clean. In a bare clone that
does not have the challenge repository's `possession_metrics`, the lab-block
check skips gracefully and the other 45 checks still pass.

**Tune set — the selection run** (6 lab matches, `train/sweep.py`, coordinate
descent over 12 parameters, 2 passes, 78 evaluated configs; picked by mean
`spell_f1_mean`, macro tie-break):

| config | mean spellF1 | mean macroF1 | alive/dead F1 | pred/gt spells |
|---|---|---|---|---|
| first-principles defaults | 0.2969 | 0.6714 | 0.835 | 1.32 |
| **frozen (chosen)** | **0.3683** | 0.6145 | 0.800 | 1.16 |

Per-match spellF1 of the frozen config: 0.4103 · 0.3430 · 0.3302 · 0.3235 ·
0.4217 · 0.3812. Selection is spell-first by design — spellF1 is the ranking
metric and macro only breaks ties. The macro/alive-dead decrease is dead-ball
recall against lab GT, which over-marks dead (§8.1): the frozen config emits
less `D` because fewer dead runs means less fragmentation, which is exactly
what spellF1 rewards.

**Held-out — 14 lab matches never seen by the sweep** (generalization of the
same choice):

| config | spellF1 | macroF1 | alive/dead F1 | poss/min (GT 3.1) | D-runs/min (GT 1.1) | D share (GT 45 %) |
|---|---|---|---|---|---|---|
| first-principles defaults | 0.2707 | 0.6491 | 0.821 | 3.8 | 2.2 | 32 % |
| **frozen** | **0.3487** | 0.5786 | 0.774 | **3.3** | **1.1** | 24 % |

+0.078 spellF1 out-of-sample, and structure moves from 2× over-splitting
(2.2 D-runs/min against GT's 1.1) to an exact D-run rate match with 3.3
possessions/min against GT's 3.1. The frozen config again gives up lab
macro/alive-dead because it marks less `D` than the lab's inflated GT wants
(§8.1) — and on the competition window the same config posts the **best
alive/dead F1 (0.9125) and best D F1 (0.5069) on the board**.

**Single public-window pipeline check** — the challenge's own
`evaluation/run_evaluation.py` against
`evaluation/gt/BAR-ATH_possession_gt_public.csv` (7,500 graded frames, 21 GT
segments, 17 possessions of mean 14.2 s), run once on the exact deliverable
file:

| rank | submission | macroF1 | spellF1 | poss | dur s | transF1 | lat_f | shareE | frag |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `baseline_smooth125` | 0.5812 | **0.4242** | 16 | 18.5 | 0.1667 | −7.7 | 0.052 | 0.81 |
| 2 | **this submission** | **0.6627** | 0.3545 | 27 | 9.9 | 0.1923 | +4.8 | 0.004 | 1.57 |
| 3 | `baseline_smooth50` | 0.6207 | 0.3256 | 26 | 10.8 | 0.0816 | +4.0 | 0.048 | 1.43 |
| 4 | `baseline_smooth25` | 0.6200 | 0.2868 | 36 | 7.6 | 0.2034 | +8.2 | 0.056 | 1.90 |
| 5 | `baseline_raw` | 0.5972 | 0.0378 | 131 | 2.0 | 0.1410 | −3.1 | 0.046 | 6.52 |

(GT reference: 17 possessions, mean 14.2 s. Ranked on spellF1; macroF1 is the
tie-break.)

Reading it:

* **The pipeline is correct**: zero abstains, home-share error 0.004 (both
  teams' shares calibrated), best alive/dead F1 on the board — alignment,
  format and label semantics confirmed.
* **spellF1 0.3545 (board #2), macroF1 0.6627 (board #1)** — above the
  full-match-winning `smooth50` (0.3256 / 0.6207) on both metrics, stated
  with the noise caveat the challenge itself preaches: with 17 GT
  possessions one missed spell moves window spellF1 ≈ 0.06 (the full-match
  noise band is 0.04), so these gaps are indicative, not significant. The
  only system ahead on spellF1 is `smooth125`, the exact trap the challenge
  documents: it wins this window (0.4242) and loses the full match (0.2242
  vs smooth50's 0.2433) by under-splitting. Nothing here was used for
  tuning — the evidence that the frozen config generalizes is the +0.078
  held-out gain above, together with the compression stress at the end of
  §7 showing this pipeline's window scores are unbiased (|Δ| ≤ 0.03).
* **27 possessions vs GT 17 (+10)** — over-segmentation is our weakness, but
  it is the same behavior, at the same magnitude, as `smooth50` (+9, 26 vs
  17) — the baseline that wins the full match — while `smooth25` (+19) and
  `raw` (+114) are far worse. Honest cost: dead recall in the window is 0.39
  (GT's four dead spells average 14.7 s; we produce six dead runs in-window
  and our restart-driven dead runs average 6.1 s full-match), so we capture a
  stoppage's core but not all of its preparation time: mean possession 9.9 s
  vs GT's 14.2 s.

**Structural self-check of the final submission (own output only, no GT)** —
461 possessions = **4.4/min** over the 105.9 graded minutes (published window
structure: 3.4/min), 197 D runs = 1.9/min, `D` share **32.0 %**, mean
possession 9.4 s. Attribution of every graded D run using only our own labels
and the provided tags: 165 restart-driven (mean 6.1 s; all 59 `SET PIECES`
windows produce ≥1 D run), 29 motion-lull (mean 35.5 s — 51 % of all D
frames), 3 ball-out (mean 1.6 s); zero minimum-dwell violations (median alive
gap between D runs 157 frames, none below the 40-frame dwell). Our own
in-window rate is 5.4 possessions/min, higher than our full-match 4.4 — the
excess segmentation is a general tendency, not a full-match artifact, and it
runs in the same direction as the full-match-winning baseline.

**Window-vs-full compression stress** (`train/window_stress.py`; lab data
only, no competition GT) — the public signal is 5 minutes, the ranking is
the full match, and the official baselines compress between those two views
of the *same* prediction (`smooth50` −0.082, `smooth125` −0.200). Measured
legally by decoding each of the 20 lab matches once and scoring the
identical prediction twice — whole match vs 5-minute slices at each half's
opening (decode cold start at the block boundary, exactly like the
competition decode) and closing (80 slices):

| slice | ΔspellF1 (window − full) | sd | ΔmacroF1 |
|---|---|---|---|
| match-open (competition analog, n=20) | **+0.033** | 0.094 | +0.023 |
| all positions (n=80) | −0.018 | 0.108 | −0.018 |

Our window numbers are essentially unbiased (|Δ| ≤ 0.03) where the baselines
lose 0.08–0.20 — the smoothing-window mismatch that makes `smooth125` the
documented trap does not apply to a decoder selected on full matches. Two
honest caveats: single 5-minute slices are noisy for any method (sd ≈ 0.10,
which is itself why a window win does not transfer), and the lab is not the
competition match — but the risk we are guarding against here, *systematic*
window-to-full compression, is measured to be near zero. Full log:
`train/window_stress_results.json`.

**Failure-mode decomposition** (`train/error_analysis.py` — the official
scorer's own spell functions, 14 held-out lab matches, frozen vs
first-principles defaults; pure diagnosis, used for nothing):

| metric (held-out) | frozen | defaults |
|---|---|---|
| spellF1 (mean per match) | **0.3487** | 0.2707 |
| spell recall @ IoU 0.3 | **0.517** | 0.457 |
| spell precision @ IoU 0.3 | **0.461** | 0.350 |
| GT spells split into ≥ 2 pieces | **7.3 %** | 10.9 % |
| spurious predicted spells (total) | **2581** | 3618 |
| missed GT spells (total) | 2057 | 2314 |
| mean duration, missed spells | 4.2 s | 4.7 s |
| mean duration, matched spells | 16.5 s | 17.6 s |
| misses inside a restart window | 37 % | 35 % |
| D-runs predicted / GT | **1610 / 1499** | 3263 / 1499 |

Two readings. First, the frozen config dominates the defaults on *every*
error mechanism, not only the headline score: +0.06 recall, +0.11
precision, 29 % fewer false possessions, 11 % fewer misses, and D-run
counts within 7 % of GT where the defaults emit 2.2× too many — the sweep
moved the machinery itself, not just the metric. Second, the characteristic
miss is a **short possession**: missed spells average 4.2 s against 16.5 s
for recovered ones, and 37 % sit inside a restart window. Some of those
"misses" are SkillCorner gap fragments our decoder deliberately holds dead
through (§8.1), so treat recall as a lower bound on true recovery. Full
log: `train/error_analysis_results.json`.

## 8. Limitations

1. **Lab GT approximates "not in play".** SkillCorner's possession model marks
   ~45 % of frames as no-possession vs the competition's ~20–25 % dead time;
   absolute dead thresholds may be biased toward more `D`. Mitigated by
   (a) relative selection (all configs see the same GT) and (b) structural
   sanity checks of the final output against published structure numbers.
2. **Phase-tag team semantics are inferred from code names.** Ambiguous codes
   are excluded; the prior is deliberately weak (`w_tag = 0.30`) and only
   biases fresh tracking votes, so a misreading cannot dominate.
3. **Restart tags in the lab are synthetic** (derived from lab GT gap ends,
   jittered): they exercise the restart code path but cannot teach real tag
   quality. The real `SET PIECES` windows are mostly fixed 35 s long
   (measured: 59 tags, median 35 s), while the lab brackets are ~5 s — so the
   lab never exercised a window staying open for half a minute. The decoder
   is protected against that mismatch by design (exit is motion/kick-driven,
   not window-close-driven: the final submission's restart D runs average
   6.1 s inside those 35 s windows), but `t_enter`'s restart path inherits
   first-principles defaults and lab motion behavior.
4. **No video model** (dropped asset) — all signal comes from tracking +
   annotations; an aerial-ball occlusion that hides the ball from tracking is
   handled only by holding the last team, not by recovering the ball from
   pixels.
5. **`w_tag` is not tunable in the lab** (no phase tags exist there) and keeps
   its first-principles default.
6. The homography is validated but unused; a future iteration could use it for
   image-space ball recovery.

**What we would fix with more time** (ordered by the measured payoff in the
§7 decomposition):

7. **Short, restart-adjacent possessions.** 37 % of missed spells sit
   inside a restart window and misses average 4.2 s. The decoder holds `D`
   until the kick and then needs sustained evidence; letting the phase
   tags' team label (the competition's phases carry `Team`, the lab's
   synthetic set-piece tags do not) seed the restart team would recover the
   first seconds after a restart without loosening the switch threshold that
   protects everything else.
8. **Dead-ball boundary alignment.** Against lab GT the D-run *count* is
   within 7 % but only 43 % match at IoU 0.3 — the boundaries, not the
   counts, are off. Exact touchline calls from the homography (currently
   parsed and dropped, §2) would tighten ball-out edges, and with them the
   restart/lull boundaries.
9. **A second legal validation source with competition-like dead-ball
   semantics.** The lab GT marks ~45 % of frames dead against the
   competition's ~20–25 % (§8.1), so lab evidence and the (one-shot)
   window signal pull the D rate in opposite directions. Another labelled
   source whose dead share matches the competition would remove that
   conflict and let the D rate be selected rather than argued.

## 9. Reproduction

From a fresh clone of this repository and the challenge's four-file data
bundle (listed in the README under **Data**):

```bat
pip install -r requirements.txt
python make_submission.py --data-dir <data-dir> --out submission.csv
```

The run is deterministic — same data + config gives a byte-identical CSV.
Verified three ways, all producing SHA256
`8CEE6FE3A39F2715F993DA595E66351BB476EDD8B302CB06A6DB262292188E99`:

1. **cold run** with the parse cache removed (rebuilt from the raw text,
   59 s), and **warm run** (~12 s) on this machine;
2. a copy of **exactly the files git tracks**, simulating a fresh clone —
   unit tests run there (45 checks + graceful skip of the 2 lab checks that
   need the challenge's scorer) and the regenerated file matches byte for
   byte.

Dependencies are pinned (`requirements.txt`), the CSV contract (`frame,label`,
199,319 rows, frames 0..N-1 in order) is validated before the file is
written, and `python tests/test_core.py` gives 47 checks / 0 failures where
the challenge repository is available. Independent format validation:
`python <challenge-repo>/evaluation/validate_submission.py submission.csv`
→ `Valid`. The submission uploaded to the form is this file under the team
name — same bytes.
