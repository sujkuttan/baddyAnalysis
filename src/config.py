import numpy as np

COURT_WIDTH = 6.1
COURT_LENGTH = 13.4

COURT_CORNERS_METERS = np.array(
    [
        [0.0, 0.0],
        [COURT_WIDTH, 0.0],
        [COURT_WIDTH, COURT_LENGTH],
        [0.0, COURT_LENGTH],
    ],
    dtype=np.float64,
)

CANONICAL_STROKES = [
    "clear",
    "smash",
    "drive",
    "drop",
    "net_shot",
    "lift",
    "push",
    "block",
    "cross_court",
    "rush",
    "long_serve",
]

STROKE_TO_ID = {s: i for i, s in enumerate(CANONICAL_STROKES)}

SYNONYMS = {
    "soft_lift_or_push": "lift",
    "defensive_lift": "lift",
    "short_serve": "long_serve",
    "net": "net_shot",
    "netshot": "net_shot",
    "net_shot": "net_shot",
    "crosscourt": "cross_court",
    "cross_court": "cross_court",
}


def canonical_stroke(name):
    if name is None:
        return None
    name = str(name).strip().lower()
    if name in STROKE_TO_ID:
        return name
    return SYNONYMS.get(name)


PLAYER_IDS = ["player_1", "player_2"]
