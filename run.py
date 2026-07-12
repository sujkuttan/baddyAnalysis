import argparse
import importlib.util
import json
import os

from src import pipeline
from src import shuttle as shuttlemod
from src import inpaintnet as inpaintmod
from src.config import remap_corners, validate_court_corners, CORNER_ORDER_CANON


DEFAULT_LOCAL_VIDEO = "/home/sujith/baddyCoach/videos/sample_5min.mp4"
DEFAULT_LOCAL_CORNERS = [[466, 77], [831, 80], [1181, 641], [148, 637]]
DEFAULT_TRACKNET = "weights/TrackNet_best.pt"
DEFAULT_INPAINTNET = "weights/InpaintNet_best.pt"
DEFAULT_LOCAL_FRAMES = 450
DEFAULT_LOCAL_BATCH_SIZE = 8
PIPELINE_REQUIRED_PACKAGES = ("ultralytics",)


def load_corners(path, order):
    if path is None:
        pts = DEFAULT_LOCAL_CORNERS
        validate_court_corners(pts)
        return pts
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        pts = data["corners"]
        order = data.get("order", order)
    else:
        with open(path) as f:
            pts = json.load(f)
    pts = remap_corners(pts, order)
    validate_court_corners(pts)
    return pts


def require_pipeline_dependencies():
    missing = [pkg for pkg in PIPELINE_REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]
    if missing:
        pkgs = ", ".join(missing)
        raise SystemExit(
            f"Missing Python package(s): {pkgs}\n"
            "Install repo dependencies with: python3 -m pip install -r requirements.txt"
        )


def cmd_pipeline(args):
    require_pipeline_dependencies()
    corners = load_corners(args.corners, args.corners_order)
    try:
        img_h, img_w = (int(x) for x in args.tracknet_img_size.split(","))
    except Exception:
        img_h, img_w = 288, 512
    if img_h % 8 or img_w % 8:
        print("WARNING: --tracknet_img_size H,W must both be divisible by 8; "
              f"got {img_h},{img_w} -> TrackNet may fail. Using 288,512.")
        img_h, img_w = 288, 512
    pipeline.run_full_pipeline(
        args.video, corners, out_dir=args.out, labels_csv=args.labels,
        device=args.device, tracknet_weights=args.tracknet,
        inpaintnet_weights=args.inpaintnet, use_mbh=args.mbh,
        llm_provider=args.llm_provider, llm_key=args.llm_key,
        max_frames=args.max_frames, batch_size=args.batch_size, debug=args.debug,
        max_players=args.max_players, pose_model=args.pose_model,
        pose_upscale=args.pose_upscale, pose_conf=args.pose_conf,
        tracknet_crop=args.tracknet_crop, far_tile=args.far_tile,
        tracknet_img_size=args.tracknet_img_size,
    )


def cmd_ab(args):
    res = pipeline.ab_compare(args.labels, args.new, args.bst)
    print("BST:", res["bst"]["accuracy"], "NEW:", res["new"]["accuracy"])


def cmd_shuttle_smoke(args):
    import cv2
    import numpy as np
    import torch

    device = args.device
    if str(device) == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    tracker = shuttlemod.TrackNetShuttle(args.tracknet, device=device)
    inpainter = inpaintmod.load_inpaintnet(args.inpaintnet, device=device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_hw = None
    coords = []
    n_read = 0
    cap_max = args.max_frames if args.max_frames is not None else float("inf")
    while n_read < cap_max:
        batch = []
        while len(batch) < args.batch_size and n_read < cap_max:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_hw is None:
                frame_hw = frame.shape[:2]
            batch.append(frame)
            n_read += 1
        if not batch:
            break
        coords.extend(tracker.predict_frames(batch))
    cap.release()

    coords = np.array(coords, dtype=np.float64)
    valid = int(np.sum(~np.isnan(coords).any(axis=1))) if len(coords) else 0
    print(f"[shuttle-smoke] frames={len(coords)} fps={fps:.2f} "
          f"tracknet_valid={valid} ({100*valid/max(len(coords),1):.1f}%)")

    if inpainter is not None and frame_hw is not None:
        repaired = inpaintmod.rectify_trajectory(
            coords, frame_hw[1], frame_hw[0], inpainter, device=device, seq_len=args.seq_len)
        repaired_valid = int(np.sum(~np.isnan(repaired).any(axis=1))) if len(repaired) else 0
        print(f"[shuttle-smoke] inpaint_valid={repaired_valid} "
              f"({100*repaired_valid/max(len(repaired),1):.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Badminton analysis pipeline")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("pipeline")
    p.add_argument("--video", default=DEFAULT_LOCAL_VIDEO)
    p.add_argument("--corners", default=None,
                   help="JSON file with court corners. Can contain {\"corners\":[...],\"order\":\"BL,BR,TL,TR\"} "
                        "or pass --corners_order. Order must map to canonical TL,TR,BR,BL. "
                        "Defaults to sample video corners.")
    p.add_argument("--corners_order", default=",".join(CORNER_ORDER_CANON),
                   help="semantic order of the 4 points in --corners, e.g. TL,TR,BR,BL or BL,BR,TL,TR")
    p.add_argument("--out", default="data")
    p.add_argument("--labels", default="labels_import.csv")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tracknet", default=DEFAULT_TRACKNET)
    p.add_argument("--inpaintnet", default=DEFAULT_INPAINTNET)
    p.add_argument("--mbh", action="store_true")
    p.add_argument("--max_frames", type=int, default=DEFAULT_LOCAL_FRAMES, help="limit frames (quick test)")
    p.add_argument("--batch_size", type=int, default=DEFAULT_LOCAL_BATCH_SIZE, help="frames per batch")
    p.add_argument("--max_players", type=int, default=None,
                   help="known player count (2=singles, 4=doubles); merges fragmented tracks into K players")
    p.add_argument("--pose_model", default="yolov8s-pose.pt",
                   help="YOLOv8 pose weights; use a larger model (e.g. yolov8m-pose.pt) "
                        "for better detection of small/distant players")
    p.add_argument("--pose_upscale", type=float, default=1.0,
                   help="upscale factor applied before pose detection (e.g. 1.5) to give "
                        "small/distant players more pixels; keypoints are rescaled back")
    p.add_argument("--pose_conf", type=float, default=0.25,
                   help="YOLO pose detection confidence threshold (default 0.25); lower "
                        "it (e.g. 0.1) to recover faint/small distant players")
    p.add_argument("--tracknet_crop", action="store_true",
                   help="crop each frame to the court region (+margins, 16:9) before the "
                        "TrackNet resize, giving the distant far-court shuttle more pixels")
    p.add_argument("--far_tile", action="store_true",
                   help="second TrackNet pass cropped to the far court (+headroom) for ~2x "
                        "pixels on the perspective-compressed far shuttle; trusted in far half")
    p.add_argument("--tracknet_img_size", type=str, default="288,512",
                   help="TrackNet input resolution 'H,W' (must be divisible by 8); "
                        "higher = more pixels on the far shuttle (Option A), more GPU cost")
    p.add_argument("--debug", action="store_true", help="print shuttle/contact/label diagnostics")
    p.add_argument("--llm_provider", default=None)
    p.add_argument("--llm_key", default=None)
    p.set_defaults(func=cmd_pipeline)

    a = sub.add_parser("ab")
    a.add_argument("--labels", default="labels_import.csv")
    a.add_argument("--new", default="data/new_predictions.csv")
    a.add_argument("--bst", default=None)
    a.set_defaults(func=cmd_ab)

    s = sub.add_parser("shuttle-smoke")
    s.add_argument("--video", required=True)
    s.add_argument("--tracknet", required=True)
    s.add_argument("--inpaintnet", default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--max_frames", type=int, default=16,
                   help="limit frames for local CPU/WSL validation")
    s.add_argument("--batch_size", type=int, default=8,
                   help="tiny local batches avoid CPU/WSL memory spikes")
    s.add_argument("--seq_len", type=int, default=16)
    s.set_defaults(func=cmd_shuttle_smoke)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
