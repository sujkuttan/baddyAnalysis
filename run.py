import argparse
import json
import os

from src import pipeline
from src.config import remap_corners, validate_court_corners, CORNER_ORDER_CANON


def load_corners(path, order):
    if path.endswith(".json"):
        data = json.load(open(path))
        pts = data["corners"]
        order = data.get("order", order)
    else:
        pts = json.load(open(path))
    pts = remap_corners(pts, order)
    validate_court_corners(pts)
    return pts


def cmd_pipeline(args):
    corners = load_corners(args.corners, args.corners_order)
    pipeline.run_full_pipeline(
        args.video, corners, out_dir=args.out, labels_csv=args.labels,
        device=args.device, tracknet_weights=args.tracknet, use_mbh=args.mbh,
        llm_provider=args.llm_provider, llm_key=args.llm_key,
        max_frames=args.max_frames, batch_size=args.batch_size, debug=args.debug,
    )


def cmd_ab(args):
    res = pipeline.ab_compare(args.labels, args.new, args.bst)
    print("BST:", res["bst"]["accuracy"], "NEW:", res["new"]["accuracy"])


def main():
    ap = argparse.ArgumentParser(description="Badminton analysis pipeline")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("pipeline")
    p.add_argument("--video", required=True)
    p.add_argument("--corners", required=True,
                   help="JSON file with court corners. Can contain {\"corners\":[...],\"order\":\"BL,BR,TL,TR\"} "
                        "or pass --corners_order. Order must map to canonical TL,TR,BR,BL.")
    p.add_argument("--corners_order", default=",".join(CORNER_ORDER_CANON),
                   help="semantic order of the 4 points in --corners, e.g. TL,TR,BR,BL or BL,BR,TL,TR")
    p.add_argument("--out", default="data")
    p.add_argument("--labels", default="labels_import.csv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tracknet", default=None)
    p.add_argument("--mbh", action="store_true")
    p.add_argument("--max_frames", type=int, default=None, help="limit frames (quick test)")
    p.add_argument("--batch_size", type=int, default=128, help="frames per batch")
    p.add_argument("--debug", action="store_true", help="print shuttle/contact/label diagnostics")
    p.add_argument("--llm_provider", default=None)
    p.add_argument("--llm_key", default=None)
    p.set_defaults(func=cmd_pipeline)

    a = sub.add_parser("ab")
    a.add_argument("--labels", default="labels_import.csv")
    a.add_argument("--new", default="data/new_predictions.csv")
    a.add_argument("--bst", default=None)
    a.set_defaults(func=cmd_ab)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
