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

CORNER_ORDER_CANON = ["TL", "TR", "BR", "BL"]


def remap_corners(points, order):
    """Reorder a list of 4 (x,y) points into canonical [TL, TR, BR, BL].

    `order` is a comma-separated string describing which semantic corner each
    input point is, e.g. "BL,BR,TL,TR" means points[0] is BL, points[1] is BR,
    points[2] is TL, points[3] is TR.
    """
    import numpy as np
    order = [o.strip().upper() for o in str(order).split(",")]
    if len(order) != 4 or sorted(order) != sorted(CORNER_ORDER_CANON):
        raise ValueError("corner order must be a permutation of TL,TR,BR,BL")
    out = [None] * 4
    for o, pt in zip(order, points):
        out[CORNER_ORDER_CANON.index(o)] = list(pt)
    return np.array(out, dtype=np.float64)


def validate_court_corners(corners):
    """Geometry sanity check for canonical [TL, TR, BR, BL] in image pixels
    (x right, y down). Raises ValueError if the quad is inverted/swapped."""
    corners = np.array(corners, dtype=np.float64)
    tl, tr, br, bl = corners
    top_y = (tl[1] + tr[1]) / 2.0
    bot_y = (bl[1] + br[1]) / 2.0
    if top_y >= bot_y:
        raise ValueError(
            "Court corners appear inverted (top edge y >= bottom edge y). "
            "Check the corner ORDER -- it is likely not TL,TR,BR,BL."
        )
    if tl[0] >= tr[0]:
        raise ValueError("TL.x >= TR.x: left/right swapped in corners.")
    if bl[0] >= br[0]:
        raise ValueError("BL.x >= BR.x: left/right swapped in corners.")
    return True

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
