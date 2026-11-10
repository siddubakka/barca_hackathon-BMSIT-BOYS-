"""Competition constants for the BAR-ATH tactical source.

All frame numbers here are *submission rows* (video frames, 0-based).
Sources: official games.py (graded blocks), FAQ (published frame ranges and
the hard-coded dead stretch before the second-half kickoff).
"""

N_VIDEO_FRAMES = 199319
FPS = 25

# games.py GAMES["BAR-ATH"]["pitch"] -- metres, measured from corrected
# tracking. Used to convert normalised 0..1 coordinates to metres.
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

# Published graded blocks, half-open (derived as halves[h] - offsets[h] in
# games.py; also stated in the FAQ). Everything outside is ungraded.
GRADED_BLOCKS = ((11185, 90652), (110808, 190198))

# FAQ: the second half is graded from 110808 but does not kick off until
# 113742 -- "those 2,934 frames are graded and all D".
SECOND_HALF_PRE_KICKOFF = (110808, 113742)

# Ungraded stretches we hard-anchor as D (ball not in play) so the decoder
# never has to reason about warm-up / halftime / post-match footage.
PRE_MATCH = (0, 11185)
HALFTIME = (90652, 110808)
POST_MATCH = (190198, N_VIDEO_FRAMES)

GAME = "BAR-ATH"
