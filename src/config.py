import numpy as np

COURT_WIDTH = 6.1
COURT_LENGTH = 13.4

# Geometric clip margin for warped shuttle coords: points further than this
# outside the court lines are treated as false positives (stands / teleports).
# Kept generous so legitimately out-of-bounds shots are not discarded.
OOB_MARGIN_M = 2.0

# Hard cap on instantaneous shuttle speed (m/s). Detections above this are
# treated as wild in-bounds teleports and nulled before contact detection.
MAX_SHUTTLE_SPEED_MPS = 100.0

# Contact cue in image space: max shuttle->wrist pixel distance to count as a
# hit. Avoids the court-homography extrapolation that pushes aerial shuttles
# off-court (and so misses real hits). Tune to frame resolution.
IMAGE_CONTACT_MAX_DIST_PX = 60.0

# Attribution gate: a detected contact is only assigned to a player if that
# player's wrist is within this distance (m) of the shuttle at the contact
# frame. Contacts farther than this from every wrist are dropped as spurious
# rather than being handed to the nearest (but wrong) player.
ATTRIB_MAX_DIST_M = 3.0

# Half-aware contact attribution: assign a hit to the player on the court half
# where the shuttle is (halves defined by stable player foot positions, not the
# noisy shuttle warp), within a generous gate. Counters the better-detected
# player stealing hits on the other half. Set False for legacy nearest-wrist.
HALF_AWARE_ATTRIB = True
HALF_AWARE_TOL_M = 1.0
HALF_AWARE_GATE_M = 4.0

# TrackNet decode heatmap threshold. Lower = more (incl. low-confidence) shuttle
# detections, at the cost of more false positives. Tuned down from 0.5 to recover
# shuttle detections at hit instants that otherwise get missed.
TRACKNET_HEAT_THRESH = 0.4

# Court-region crop for TrackNet: crop each frame to the court bbox (+ margins,
# grown to the model's 16:9 aspect) before the 512x288 resize, so the distant
# far-court shuttle gets more effective pixels. OFF by default -- enable for an
# A/B run once it is validated to help (small/far shuttle recovery). Margins are
# fractions of court width (left/right) and court height (top/bottom); top is
# large to keep aerial clears (which rise above the far baseline) inside the crop.
# NOTE: only helps when the court does NOT already fill the frame. For a tightly
# framed view (court ~= whole frame) the crop collapses to the full frame and is
# a no-op; the far-court shuttle is small from perspective, not wasted border --
# use a far-half tile (Phase C) or a larger img_size in that case.
TRACKNET_COURT_CROP = False
TRACKNET_CROP_MARGINS = {"left": 0.15, "right": 0.15, "top": 0.25, "bottom": 0.1}

# Image-space shuttle velocity gate: null detections whose pixel displacement
# from a robust local (median-of-neighbors) estimate exceeds this many px/frame.
# Removes false teleport detections (on players/background) that warp far
# off-court, while keeping smoothly-moving real (incl. aerial) detections.
SHUTTLE_IMG_MAX_STEP_PX = 250.0

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
